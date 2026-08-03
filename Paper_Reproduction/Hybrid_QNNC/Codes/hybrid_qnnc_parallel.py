
import os
import time
import glob

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.circuit.library import QNNCircuit

from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Globals (same as your original notebook, plus parallelization knobs)
# =============================================================================
root_folder = 'QNNC_hybrid'

torch.manual_seed(42)
np.random.seed(42)

NUM_FEATURES = 3
NUM_QUBITS = NUM_FEATURES
NUM_TARGETS = 1

FEATURE_MAP_REPS_LIST = [1]
ANSATZ_REPS_LIST = [1]
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']

LEARNING_RATE = 0.01
BATCH_SIZE = 30
NUM_EPOCHS = 100

N_REPEATS = 1
TEST_SIZE = 1  # LOOCV

CLASSIFIER_THRESHOLD = 19

date = '06_08_25_1'
dataset_name = "qml_training-validation-data.csv"

# --- Parallelization knobs ---
N_JOBS = 20          # how many CPU cores / worker processes to use
USE_GPU = False      # keep False when N_JOBS > 1 -- see note above
VERBOSE_EPOCHS = False  # set True to print every epoch's loss (very noisy at scale)

DEVICE = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

RESULT_DIR = f'{root_folder}/result'
FOLD_DIR = f'{RESULT_DIR}/folds'
LOG_DIR = f'{root_folder}/logs'

for d in (RESULT_DIR, FOLD_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Unchanged scientific functions (copied from your notebook)
# =============================================================================
def get_qnn_torch_model(entangle, feature_map_reps, ansatz_reps):
    input_params = ParameterVector("x", NUM_FEATURES)

    feature_map_template = PauliFeatureMap(
        feature_dimension=NUM_FEATURES,
        reps=feature_map_reps,
        entanglement=entangle
    )
    feature_map = feature_map_template.assign_parameters(input_params)

    ansatz_template = RealAmplitudes(NUM_QUBITS, reps=ansatz_reps, entanglement=entangle)
    num_ansatz_params = ansatz_template.num_parameters
    weight_params = ParameterVector("\u03b8", num_ansatz_params)
    ansatz = ansatz_template.assign_parameters(weight_params)

    qc = QNNCircuit(
        feature_map=feature_map,
        ansatz=ansatz_template,
    )

    parity = lambda x: "{:b}".format(x).count("1") % 2
    output_shape = 2

    sampler = Sampler()

    qnn = SamplerQNN(
        circuit=qc,
        interpret=parity,
        output_shape=output_shape,
        sampler=sampler,
        sparse=False,
        input_gradients=False,
    )

    initial_weights = 0.01 * (2 * np.random.rand(qnn.num_weights) - 1)
    qnn_torch_model = TorchConnector(qnn, initial_weights=torch.tensor(initial_weights, dtype=torch.float32))

    return qnn_torch_model.to(DEVICE)


class HybridModel(nn.Module):
    def __init__(self, qnn_model):
        super().__init__()
        self.qnn = qnn_model

    def forward(self, x):
        return self.qnn(x)


def prepare_dataset_k_fold(X, y, train_indices, test_indices):
    X_train_raw, X_test_raw = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    element_test = X_test_raw[:, 0]
    element_train = X_train_raw[:, 0]

    X_train = X_train_raw[:, 1:]
    X_test = X_test_raw[:, 1:]

    full_X = np.vstack([X_train, X_test])

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(full_X)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, element_test, element_train


# =============================================================================
# Per-fold worker: runs the full hyperparameter sweep for ONE fold
# =============================================================================
def process_fold(fold_id, train_indices, test_indices, X, y):
    """
    Trains every (entanglement, feature_map_reps, ansatz_reps) combo for a
    single fold, writes a checkpoint CSV, and returns its path.

    If the checkpoint already exists (from a previous run), skips the fold
    entirely -- this is what gives us resume support.
    """
    checkpoint_path = f'{FOLD_DIR}/fold_{fold_id:04d}.csv'
    if os.path.exists(checkpoint_path):
        return checkpoint_path  # already done -- resume support

    # avoid PyTorch spawning its own thread pool inside each of the N_JOBS
    # worker processes -- would badly oversubscribe the CPU otherwise
    torch.set_num_threads(1)

    LOSS = nn.CrossEntropyLoss()

    X_train, y_train, X_test, y_test, element_test, element_train = \
        prepare_dataset_k_fold(X, y, train_indices, test_indices)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    rows = []

    for entanglement in ENTANGLEMENT_LIST:
        for feature_map_reps in FEATURE_MAP_REPS_LIST:
            for ansatz_reps in ANSATZ_REPS_LIST:

                model = HybridModel(get_qnn_torch_model(
                    entangle=entanglement,
                    feature_map_reps=feature_map_reps,
                    ansatz_reps=ansatz_reps,
                )).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    running_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = LOSS(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * batch_X.size(0)
                    epoch_loss = running_loss / len(train_loader.dataset)

                    model.eval()
                    test_loss = 0.0
                    with torch.no_grad():
                        for batch_X_test, batch_y_test in test_loader:
                            batch_X_test = batch_X_test.to(DEVICE)
                            batch_y_test = batch_y_test.to(DEVICE)
                            outputs_test = model(batch_X_test)
                            loss_test = LOSS(outputs_test, batch_y_test)
                            test_loss += loss_test.item() * batch_X_test.size(0)
                    epoch_test_loss = test_loss / len(test_loader.dataset)

                    if VERBOSE_EPOCHS:
                        print(f"[fold {fold_id}] {entanglement}/fmr{feature_map_reps}/ar{ansatz_reps} "
                              f"epoch {epoch + 1}/{NUM_EPOCHS} "
                              f"train_loss={epoch_loss:.4f} test_loss={epoch_test_loss:.4f}")

                # --- evaluation ---
                model.eval()
                all_preds, all_targets = [], []
                all_preds_train, all_targets_train = [], []
                with torch.no_grad():
                    for batch_X_test, batch_y_test in test_loader:
                        batch_X_test = batch_X_test.to(DEVICE)
                        outputs_test = model(batch_X_test)
                        all_preds.extend(outputs_test.cpu().numpy())
                        all_targets.extend(batch_y_test.cpu().numpy())
                    for batch_X_train, batch_y_train in train_loader:
                        batch_X_train = batch_X_train.to(DEVICE)
                        outputs_train = model(batch_X_train)
                        all_preds_train.extend(outputs_train.cpu().numpy())
                        all_targets_train.extend(batch_y_train.cpu().numpy())

                all_preds = np.array([1 if item[1] > item[0] else 0 for item in np.array(all_preds)])
                all_targets = np.array(all_targets)
                all_preds_train = np.array([1 if item[1] > item[0] else 0 for item in np.array(all_preds_train)])
                all_targets_train = np.array(all_targets_train)

                if USE_GPU and torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated()
                    gpu_mem_max = torch.cuda.max_memory_allocated()
                else:
                    gpu_mem = gpu_mem_max = None  # no CUDA -- avoid crashing on CPU-only machines

                new_row = {
                    'fold_id': fold_id,
                    'entanglement': entanglement,
                    'feature_map_reps': feature_map_reps,
                    'ansatz_reps': ansatz_reps,
                    'element test': element_test,
                    'actual test': all_targets.flatten(),
                    'predicted test': all_preds.flatten(),
                    'element train': element_train,
                    'actual train': all_targets_train.flatten(),
                    'predicted train': all_preds_train.flatten(),
                    'final_train_loss': epoch_loss,
                    'final_test_loss': epoch_test_loss,
                    'gpu_mem_allocated': gpu_mem,
                    'gpu_mem_max_allocated': gpu_mem_max,
                }
                rows.append(new_row)

    fold_df = pd.DataFrame(rows)
    tmp_path = checkpoint_path + '.tmp'
    with np.printoptions(linewidth=10000):
        fold_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, checkpoint_path)

    return checkpoint_path


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':

    def _name(lst):
        return lst[0] if len(lst) == 1 else lst

    FEATURE_MAP_REPS_LIST_NAME = _name(FEATURE_MAP_REPS_LIST)
    ANSATZ_REPS_LIST_NAME = _name(ANSATZ_REPS_LIST)
    ENTANGLEMENT_LIST_NAME = _name(ENTANGLEMENT_LIST)

    file_name = (f'{RESULT_DIR}/FMR_{FEATURE_MAP_REPS_LIST_NAME}_AR_{ANSATZ_REPS_LIST_NAME}'
                 f'_E_{ENTANGLEMENT_LIST_NAME}_{date}.csv')
    print('Final results will be merged into:', file_name)
    print(f'Device for training: {DEVICE} (USE_GPU={USE_GPU})')

    # ---- load data ----
    df = pd.read_csv(dataset_name)
    X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
    y = df['SFE/mJm^-3'].values
    print('Dataset shape:', df.shape)

    # ---- regression -> classification ----
    for i in range(len(y)):
        y[i] = 0 if y[i] > CLASSIFIER_THRESHOLD else 1

    # ---- build folds ----
    print('Total number of data:', X.shape[0])
    rkf = RepeatedKFold(n_splits=X.shape[0] // TEST_SIZE, n_repeats=N_REPEATS)
    print(rkf)

    folds = list(enumerate(rkf.split(X)))
    n_folds = len(folds)
    n_combos = len(ENTANGLEMENT_LIST) * len(FEATURE_MAP_REPS_LIST) * len(ANSATZ_REPS_LIST)
    print(f'{n_folds} folds x {n_combos} hyperparameter combos '
          f'= {n_folds * n_combos} total model trainings')

    already_done = sorted(glob.glob(f'{FOLD_DIR}/fold_*.csv'))
    if already_done:
        print(f'Resuming: {len(already_done)}/{n_folds} folds already have checkpoints and will be skipped.')

    # ---- run folds in parallel, N_JOBS at a time, streaming results live ----
    start = time.time()
    n_done = len(already_done)
    live_header_written = os.path.exists(file_name)

    if already_done and not live_header_written:
        seed = pd.concat((pd.read_csv(f) for f in already_done), ignore_index=True)
        seed.to_csv(file_name, index=False)
        live_header_written = True

    print(f'\n--- Dispatching {n_folds - n_done} remaining folds across {N_JOBS} workers ---\n')

    parallel = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator_unordered')
    jobs = (
        delayed(process_fold)(fold_id, train_idx, test_idx, X, y)
        for fold_id, (train_idx, test_idx) in folds
    )

    for checkpoint_path in parallel(jobs):
        n_done += 1
        fold_df = pd.read_csv(checkpoint_path)
        mean_train_loss = fold_df['final_train_loss'].mean()
        mean_test_loss = fold_df['final_test_loss'].mean()
        elapsed = time.time() - start
        rate = (n_done - len(already_done)) / elapsed if elapsed > 0 else 0
        eta_min = ((n_folds - n_done) / rate / 60) if rate > 0 else float('inf')

        print(f'[{n_done}/{n_folds}] {os.path.basename(checkpoint_path)} done | '
              f'mean train_loss={mean_train_loss:.3f} mean test_loss={mean_test_loss:.3f} | '
              f'elapsed={elapsed/60:.1f}m | ETA~{eta_min:.1f}m', flush=True)

        fold_df.to_csv(file_name, mode='a', index=False, header=not live_header_written)
        live_header_written = True

    total_elapsed = time.time() - start
    print(f'\nAll folds complete in {total_elapsed/60:.1f} minutes.')
    print(f'Results (growing live throughout the run) are in: {file_name}')

    final_check = pd.read_csv(file_name)
    print(f'Final shape: {final_check.shape}')
