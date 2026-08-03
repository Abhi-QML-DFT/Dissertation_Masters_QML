"""
Parallelized hybrid quantum-classical NN REGRESSOR (EstimatorQNN +
TorchConnector) training with repeated k-fold cross-validation.

Same design as vqc_parallel.py / hybrid_qnnc_parallel.py:
    - Parallelized ONE FOLD AT A TIME across N_JOBS worker processes.
    - Each worker runs the full (entanglement x feature_map_reps x
      ansatz_reps) hyperparameter sweep for its fold, sequentially.
    - Each fold writes its own checkpoint CSV before returning
      -> resume support: re-running the script skips folds already done.
    - Results stream into the final CSV live, fold by fold, instead of only
      being written once everything finishes.

Changes vs. the original hybrid_QNNR notebook
------------------------------------------------
1. Training loop is now wrapped in `process_fold`, one call per fold,
   dispatched via joblib.Parallel instead of a single sequential `for`
   loop over every fold x combo.
2. Per-epoch print statements are off by default (VERBOSE_EPOCHS) --
   with many folds x combos x epochs running across N_JOBS workers
   simultaneously, per-epoch prints would flood the console. A single
   summary line prints per completed fold instead.
3. torch.set_num_threads(1) is set inside each worker so PyTorch doesn't
   spawn its own thread pool on top of joblib's process pool (thread
   oversubscription across N_JOBS processes would badly hurt throughput).
4. Your original notebook never moved tensors/model to a CUDA device (no
   `device` variable existed at all -- everything ran on CPU already), so
   this script keeps that as-is: pure CPU, which is exactly right for
   N_JOBS-way process parallelism anyway.

Usage
-----
    python hybrid_qnnr_parallel.py

Re-running after an interruption automatically skips folds whose checkpoint
CSV already exists in QNNR_hybrid/result/folds/.
"""

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
from sklearn.metrics import mean_squared_error, r2_score

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp

from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Globals (same as your original notebook, plus parallelization knobs)
# =============================================================================
root_folder = 'QNNR_hybrid'

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

date = '24_19_25_1'
dataset_name = "qml_training-validation-data.csv"

# --- Parallelization knobs ---
N_JOBS = 20             # how many CPU cores / worker processes to use
VERBOSE_EPOCHS = False  # set True to print every epoch's loss (very noisy at scale)

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

    qc = QuantumCircuit(NUM_QUBITS)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (NUM_QUBITS - 1), 1.0)])

    estimator = Estimator()

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        input_params=input_params,
        weight_params=weight_params,
        observables=observable,
        input_gradients=False,
    )

    initial_weights = 0.01 * (2 * np.random.rand(qnn.num_weights) - 1)
    qnn_torch_model = TorchConnector(qnn, initial_weights=torch.tensor(initial_weights, dtype=torch.float32))

    return qnn_torch_model


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
def process_fold(fold_id, train_indices, test_indices, X, y, y_scaler):
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

    LOSS = nn.HuberLoss()

    X_train, y_train, X_test, y_test, element_test, element_train = \
        prepare_dataset_k_fold(X, y, train_indices, test_indices)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

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
                ))
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    running_loss = 0.0
                    for batch_X, batch_y in train_loader:
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
                        outputs_test = model(batch_X_test)
                        all_preds.extend(outputs_test.cpu().numpy())
                        all_targets.extend(batch_y_test.cpu().numpy())
                    for batch_X_train, batch_y_train in train_loader:
                        outputs_train = model(batch_X_train)
                        all_preds_train.extend(outputs_train.cpu().numpy())
                        all_targets_train.extend(batch_y_train.cpu().numpy())

                all_preds = y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))
                all_targets = y_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1))
                all_preds_train = y_scaler.inverse_transform(np.array(all_preds_train).reshape(-1, 1))
                all_targets_train = y_scaler.inverse_transform(np.array(all_targets_train).reshape(-1, 1))

                final_mse = mean_squared_error(all_targets, all_preds)
                final_r2 = r2_score(all_targets, all_preds)
                final_train_mse = mean_squared_error(all_targets_train, all_preds_train)
                final_train_r2 = r2_score(all_targets_train, all_preds_train)

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
                    'MSE test': final_mse,
                    'R2 test': final_r2,
                    'MSE train': final_train_mse,
                    'R2 train': final_train_r2,
                    'final_train_loss': epoch_loss,
                    'final_test_loss': epoch_test_loss,
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

    # ---- load data ----
    df = pd.read_csv(dataset_name)
    X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
    y = df['SFE/mJm^-3'].values
    print('Dataset shape:', df.shape)

    # ---- scale target (regression -- no classification threshold here) ----
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1))

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
        delayed(process_fold)(fold_id, train_idx, test_idx, X, y, y_scaler)
        for fold_id, (train_idx, test_idx) in folds
    )

    for checkpoint_path in parallel(jobs):
        n_done += 1
        fold_df = pd.read_csv(checkpoint_path)
        mean_r2_test = fold_df['R2 test'].mean()
        mean_r2_train = fold_df['R2 train'].mean()
        elapsed = time.time() - start
        rate = (n_done - len(already_done)) / elapsed if elapsed > 0 else 0
        eta_min = ((n_folds - n_done) / rate / 60) if rate > 0 else float('inf')

        print(f'[{n_done}/{n_folds}] {os.path.basename(checkpoint_path)} done | '
              f'mean R2(test)={mean_r2_test:.3f} mean R2(train)={mean_r2_train:.3f} | '
              f'elapsed={elapsed/60:.1f}m | ETA~{eta_min:.1f}m', flush=True)

        fold_df.to_csv(file_name, mode='a', index=False, header=not live_header_written)
        live_header_written = True

    total_elapsed = time.time() - start
    print(f'\nAll folds complete in {total_elapsed/60:.1f} minutes.')
    print(f'Results (growing live throughout the run) are in: {file_name}')

    final_check = pd.read_csv(file_name)
    print(f'Final shape: {final_check.shape}')
