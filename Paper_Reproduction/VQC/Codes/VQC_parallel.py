"""
Parallelized VQC hyperparameter sweep with repeated k-fold cross-validation.

Design
------
Parallelization happens ONE FOLD AT A TIME (not one hyperparameter-combo at a
time). Each of the 20 workers is handed a full fold; inside that fold it runs
all hyperparameter combinations sequentially and writes ONE checkpoint CSV for
that fold before returning. This keeps scheduling overhead low, keeps memory
per worker bounded, and means a crash/reboot only costs you the folds that
were mid-flight -- everything already written to disk is skipped on the next
run (resume support).

Bug fix vs. the original notebook
----------------------------------
In the original notebook, the hyperparameter sweep (`for if_pauli_feature_map
in ...`) was NOT indented inside the `for train_indices, test_indices in
rkf.split(X):` loop. That meant the fold loop ran to completion doing nothing
but variable assignment, and the hyperparameter sweep then ran exactly once,
using only the *last* fold. This script fixes that: the sweep now runs once
per fold, for every fold.

Usage
-----
    python vqc_parallel.py

Re-running after an interruption automatically skips folds whose checkpoint
CSV already exists in VQC/result/folds/.
"""

import os
import time
import glob

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import L_BFGS_B

from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional -- falls back to plain prints
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Globals (same as your original notebook)
# =============================================================================
root_folder = 'VQC'
np.random.seed(42)

NUM_FEATURES = 3

IF_PAULI_FEATURE_MAP_LIST = [True, False]
FEATURE_MAP_REPS_LIST = [1]
ANSATZ_REPS_LIST = [1]
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']

LOSS_FUNCTION = 'cross_entropy'

N_REPEATS = 1
TEST_SIZE = 1

CLASSIFIER_THRESHOLD = 19

date = '05_31_25_0'
dataset_name = "/home/ashok/Desktop/ABHI/Learning,Reproducing/qml_training-validation-data.csv"

# How many CPU cores to use. Your box has 20 physical cores.
N_JOBS = 20

# Where per-fold checkpoints and the final merged CSV go
RESULT_DIR = f'{root_folder}/result'
FOLD_DIR = f'{RESULT_DIR}/folds'
LOG_DIR = f'{root_folder}/logs'

for d in (RESULT_DIR, FOLD_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Unchanged scientific functions (copied verbatim from your notebook)
# =============================================================================
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


def reconfig_quantum_kernel_vqc(if_pauli_feature_map, feature_reps, ansatz_reps, entangle, objective_func_vals):
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)

    if if_pauli_feature_map:
        feature_map = PauliFeatureMap(feature_dimension=NUM_FEATURES, reps=feature_reps, entanglement=entangle)
    else:
        feature_map = ZZFeatureMap(feature_dimension=NUM_FEATURES, reps=feature_reps, entanglement=entangle)

    ansatz = RealAmplitudes(num_qubits=NUM_FEATURES, reps=ansatz_reps)
    optimizer = L_BFGS_B(ftol=0.000001)

    return VQC(feature_map=feature_map,
               ansatz=ansatz,
               optimizer=optimizer,
               callback=callback_graph,
               loss=LOSS_FUNCTION,
               )


def train_vqc(vqc, X_train, y_train, X_test):
    vqc.fit(X_train, np.concatenate(y_train))
    return vqc.predict(X_train), vqc.predict(X_test)


# =============================================================================
# Per-fold worker: runs ALL hyperparameter combos for ONE fold, sequentially
# =============================================================================
def process_fold(fold_id, train_indices, test_indices, X, y, y_scaler):
    """
    Runs the full hyperparameter sweep for a single fold and writes a
    checkpoint CSV for that fold. Returns the checkpoint file path.

    If the checkpoint already exists (from a previous run), the fold is
    skipped entirely -- this is what gives us resume support.
    """
    checkpoint_path = f'{FOLD_DIR}/fold_{fold_id:04d}.csv'
    if os.path.exists(checkpoint_path):
        return checkpoint_path  # already done -- resume support

    X_train, y_train, X_test, y_test, element_test, element_train = \
        prepare_dataset_k_fold(X, y, train_indices, test_indices)

    rows = []

    for if_pauli_feature_map in IF_PAULI_FEATURE_MAP_LIST:
        for feature_map_reps in FEATURE_MAP_REPS_LIST:
            for ansatz_reps in ANSATZ_REPS_LIST:
                for entanglement in ENTANGLEMENT_LIST:

                    feature_map_name = 'Pauli' if if_pauli_feature_map else 'ZZ'

                    objective_func_vals = []
                    vqc = reconfig_quantum_kernel_vqc(
                        if_pauli_feature_map=if_pauli_feature_map,
                        feature_reps=feature_map_reps,
                        ansatz_reps=ansatz_reps,
                        entangle=entanglement,
                        objective_func_vals=objective_func_vals,
                    )

                    predict_train, predict_test = train_vqc(vqc, X_train, y_train, X_test)

                    all_preds = y_scaler.inverse_transform(np.array(predict_test).reshape(-1, 1))
                    all_targets = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

                    all_preds_train = y_scaler.inverse_transform(np.array(predict_train).reshape(-1, 1))
                    all_targets_train = y_scaler.inverse_transform(np.array(y_train).reshape(-1, 1))

                    new_row = {
                        'fold_id': fold_id,
                        'feature_map_name': feature_map_name,
                        'feature_map_reps': feature_map_reps,
                        'ansatz_reps': ansatz_reps,
                        'entanglement': entanglement,
                        'element test': element_test,
                        'actual test': np.array(all_targets).flatten(),
                        'predicted test': np.array(all_preds).flatten(),
                        'element train': element_train,
                        'actual train': np.array(all_targets_train).flatten(),
                        'predicted train': np.array(all_preds_train).flatten(),
                        'R2 train': r2_score(y_train, predict_train),
                    }
                    rows.append(new_row)

    fold_df = pd.DataFrame(rows)
    # write atomically (write to tmp, then rename) so a crash mid-write
    # can't leave a half-written checkpoint that looks "done"
    tmp_path = checkpoint_path + '.tmp'
    with np.printoptions(linewidth=10000):
        fold_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, checkpoint_path)

    return checkpoint_path


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':

    # ---- output filename (same naming scheme as your original notebook) ----
    def _name(lst):
        return lst[0] if len(lst) == 1 else lst

    FEATURE_MAP_REPS_LIST_NAME = _name(FEATURE_MAP_REPS_LIST)
    ANSATZ_REPS_LIST_NAME = _name(ANSATZ_REPS_LIST)
    ENTANGLEMENT_LIST_NAME = _name(ENTANGLEMENT_LIST)
    IF_PAULI_FEATURE_MAP_LIST_NAME = str(_name(IF_PAULI_FEATURE_MAP_LIST)) \
        .replace('False', 'ZZ').replace('True', 'Pauli')

    file_name = (f'{RESULT_DIR}/FMR_{FEATURE_MAP_REPS_LIST_NAME}_AR_{ANSATZ_REPS_LIST_NAME}'
                 f'_E_{ENTANGLEMENT_LIST_NAME}_P_{IF_PAULI_FEATURE_MAP_LIST_NAME}_{date}.csv')
    print('Final results will be merged into:', file_name)

    # ---- load data ----
    df = pd.read_csv(dataset_name)
    X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
    y = df['SFE/mJm^-3'].values
    print('Dataset shape:', df.shape)

    # ---- regression -> classification ----
    for i in range(len(y)):
        y[i] = 0 if y[i] > CLASSIFIER_THRESHOLD else 1

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    # ---- build folds ----
    print('Total number of data:', X.shape[0])
    rkf = RepeatedKFold(n_splits=X.shape[0] // TEST_SIZE, n_repeats=N_REPEATS)
    print(rkf)

    folds = list(enumerate(rkf.split(X)))
    n_folds = len(folds)
    n_combos = (len(IF_PAULI_FEATURE_MAP_LIST) * len(FEATURE_MAP_REPS_LIST)
                * len(ANSATZ_REPS_LIST) * len(ENTANGLEMENT_LIST))
    print(f'{n_folds} folds x {n_combos} hyperparameter combos '
          f'= {n_folds * n_combos} total experiments')

    already_done = len(glob.glob(f'{FOLD_DIR}/fold_*.csv'))
    if already_done:
        print(f'Resuming: {already_done}/{n_folds} folds already have checkpoints and will be skipped.')

    # ---- run folds in parallel, N_JOBS at a time ----
    start = time.time()

    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
        delayed(process_fold)(fold_id, train_idx, test_idx, X, y, y_scaler)
        for fold_id, (train_idx, test_idx) in tqdm(folds, desc='Dispatching folds')
    )

    elapsed = time.time() - start
    print(f'\nAll folds complete in {elapsed/60:.1f} minutes.')

    # ---- merge all per-fold checkpoints into the final CSV ----
    fold_files = sorted(glob.glob(f'{FOLD_DIR}/fold_*.csv'))
    merged = pd.concat((pd.read_csv(f) for f in fold_files), ignore_index=True)
    merged.at[0, 'info'] = f'DATASET: {dataset_name}, LOSS: {LOSS_FUNCTION}, CLASSIFIER_THRESHOLD = {CLASSIFIER_THRESHOLD}'
    merged.to_csv(file_name, index=False)

    print(f'Merged {len(fold_files)} fold checkpoints -> {file_name}')
    print(f'Final shape: {merged.shape}')
