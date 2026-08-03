# Import Statements and Libraries (QSVR Notebook)

This section imports all the Python libraries required for the notebook.

Some libraries are general Python utilities, some belong to classical machine learning, and others are specific to Quantum Machine Learning (QML).

Although the majority of the imports are identical to those in the QSVC notebook, the learning algorithm changes from **QSVC (classification)** to **QSVR (regression)**.

---

# System Utilities

## `sys`

```python
import sys
```

`sys` is a built-in Python library that allows Python programs to communicate with the operating system.

It provides information such as:

- Python version
- Command-line arguments
- System paths
- Memory-related information

Example:

```python
import sys
print(sys.version)
```

Output:

```text
Python 3.11.4
```

### Research Remark

The notebook imports `sys`, but it is not heavily used in the Quantum Machine Learning workflow.

---

## `getopt`

```python
import getopt
```

`getopt` is used to read command-line arguments.

Example:

```text
python train.py --epochs 50
```

or

```text
python QSVR.py -e linear -f 3 -r 10
```

`getopt` interprets options such as:

- `-e`
- `-f`
- `-r`

and converts them into variables inside the program.

### Research Remark

This is mainly useful when running many experiments from the Linux terminal or an HPC cluster.

It does **not** change the machine learning algorithm.

---

## `os`

```python
import os
```

`os` stands for **Operating System**.

It allows Python to interact with files and folders.

Common examples include:

```python
os.getcwd()
```

Returns the current working directory.

```python
os.listdir()
```

Lists all files in the current directory.

```python
os.mkdir("Results")
```

Creates a new folder.

### Research Remark

Machine learning projects usually involve datasets, models, figures, and log files.

For this reason, `os` is almost always imported to manage files and directories.

---

# Numerical Computing

## NumPy

```python
import numpy as np
```

NumPy is the fundamental numerical computing library in Python.

It provides:

- multidimensional arrays,
- fast mathematical operations,
- linear algebra,
- matrix computations,
- random number generation.

Almost every machine learning library internally relies on NumPy arrays.

---

# Data Handling

## Pandas

```python
import pandas as pd
```

Pandas is the standard library for handling datasets.

It allows researchers to:

- read CSV files,
- manipulate tables,
- clean datasets,
- select columns,
- filter rows,
- prepare data for machine learning.

Example:

```python
df = pd.read_csv("dataset.csv")
```

---

# Data Visualization

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Matplotlib is used for plotting graphs.

Typical figures include:

- prediction plots,
- error curves,
- scatter plots,
- histograms,
- performance comparisons.

Visualization is essential for analysing machine learning results.

---

# Classical Machine Learning Libraries

Although this notebook implements **Quantum Machine Learning**, it still depends heavily on classical machine learning tools from **Scikit-learn**.

Quantum models usually replace only one part of the workflow—the kernel or the neural network—while the remaining pipeline is classical.

---

## Repeated K-Fold Cross Validation

```python
from sklearn.model_selection import RepeatedKFold
```

`RepeatedKFold` creates repeated train-test splits.

Instead of evaluating the model only once,

the experiment is repeated multiple times using different random partitions.

This produces a more reliable estimate of model performance.

This is particularly valuable in materials science,

where datasets are often small and different train-test splits can produce noticeably different results.

---

## Regression Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score
```

Unlike QSVC,

QSVR performs **regression** rather than classification.

Therefore, regression metrics are appropriate.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

```text
Smaller MSE

↓

Better Predictions
```

Large errors receive a greater penalty because the errors are squared.

---

### R² Score

Measures how well the regression model explains the variation in the data.

```text
R² = 1

↓

Perfect Prediction
```

```text
R² = 0

↓

No Better Than Predicting the Mean
```

Higher values indicate better regression performance.

Unlike the QSVC notebook, **R² is completely appropriate here because the target variable is continuous.**

---

# Feature Scaling

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

Machine learning algorithms often perform much better when features have similar numerical ranges.

This notebook imports two common scaling methods.

---

## MinMaxScaler

```python
MinMaxScaler
```

Transforms each feature into a fixed interval,

usually

```text
0 → 1
```

or, in this notebook,

```text
-1 → 1
```

Example:

```text
Original

5

15

25
```

↓

Scaled

```text
-1

0

1
```

### Why Use It?

It keeps every feature within the same numerical range,

preventing variables with large magnitudes from dominating the calculations.

---

## StandardScaler

```python
StandardScaler
```

Standardizes every feature so that

```text
Mean = 0

Standard Deviation = 1
```

Many machine learning algorithms,

especially Support Vector Machines,

perform well with standardized inputs.

---

### Important Note

The author's comment states

```python
# StandardScaler is sensitive to outlier
```

This is because the mean and standard deviation are strongly influenced by extreme values.

If the dataset contains large outliers,

the scaled values may become less representative of the majority of the data.

---

# Quantum Machine Learning Libraries

This is where the notebook differs from ordinary machine learning.

---

## ZZFeatureMap

```python
from qiskit.circuit.library import ZZFeatureMap
```

A feature map answers the question:

> **How do we convert ordinary numerical data into a quantum state?**

The `ZZFeatureMap` is one of Qiskit's predefined encoding circuits.

It performs:

- single-qubit rotations,
- entangling operations,
- nonlinear encoding of classical features.

This allows classical material properties such as

- electronegativity,
- bulk modulus,
- atomic volume

to be represented as quantum states.

The feature map is one of the most important components of quantum kernel methods.

---

## QSVR

```python
from qiskit_machine_learning.algorithms import QSVR
```

This is the main machine learning model used in the notebook.

**QSVR** stands for:

> **Quantum Support Vector Regressor**

It is the regression counterpart of QSVC.

Whereas QSVC predicts **discrete classes**, QSVR predicts **continuous numerical values**.

For this project,

the model predicts:

```text
Electronegativity

+

Bulk Modulus

+

Atomic Volume

↓

Stacking Fault Energy (SFE)
```

Unlike QSVC,

the goal is **not** to classify alloys into low or high SFE.

Instead,

QSVR estimates the **actual stacking fault energy value**.

---

## Fidelity Quantum Kernel

```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
```

This component computes the **quantum kernel**.

A kernel measures the similarity between two samples.

Classical Support Vector Regression typically uses kernels such as:

- Linear
- Polynomial
- Radial Basis Function (RBF)

QSVR replaces those mathematical kernels with a **quantum kernel**.

The process is:

```text
Classical Data

↓

Encode into Quantum States

↓

Compute State Fidelity

↓

Quantum Kernel Matrix
```

State fidelity measures how similar two quantum states are.

Higher fidelity indicates greater similarity.

This quantum-derived similarity matrix is then supplied to the classical Support Vector Regression algorithm.

---

# QSVC vs QSVR

Although the notebooks look almost identical,

their objectives are fundamentally different.

| QSVC | QSVR |
|------|------|
| Classification | Regression |
| Predicts classes | Predicts continuous values |
| Output: 0 or 1 | Output: Numerical SFE |
| Uses classification metrics | Uses regression metrics |
| Binary decision boundary | Continuous function approximation |

The quantum components remain almost identical:

```text
Classical Features

↓

ZZFeatureMap

↓

Quantum States

↓

Fidelity Quantum Kernel
```

The difference lies in the final learning algorithm:

```text
QSVC

↓

Classification
```

versus

```text
QSVR

↓

Regression
```

---

# Research Insight

One of the most interesting aspects of Qiskit's quantum kernel methods is that **the quantum encoding and kernel computation remain the same for both QSVC and QSVR**. The principal difference is the classical algorithm that consumes the quantum kernel matrix. In QSVC, the kernel is used by a Support Vector Classifier to separate data into discrete classes, whereas in QSVR, it is used by a Support Vector Regressor to predict continuous values. This illustrates the hybrid nature of quantum kernel methods: the quantum computer provides a novel similarity measure, while the optimization and prediction framework remain classical.

---

# Key Takeaways

- Most imports are identical to the QSVC notebook because both use the same hybrid quantum-classical workflow.
- `RepeatedKFold` creates reliable train-test splits for small materials-science datasets.
- `MinMaxScaler` and `StandardScaler` normalize the input features before training.
- `ZZFeatureMap` encodes classical material properties into quantum states.
- `FidelityQuantumKernel` computes similarities between quantum states using state fidelity.
- **QSVR (Quantum Support Vector Regressor)** predicts continuous numerical values, making it suitable for estimating stacking fault energy directly.
- Unlike QSVC, **regression metrics such as MSE and R² are appropriate and meaningful** because the target variable remains continuous.



# Global Configuration and Hyperparameters (QSVR Notebook)

This section defines the **global variables** used throughout the notebook.

These variables determine:

- where the experiment results are stored,
- how the Quantum Support Vector Regressor (QSVR) is constructed,
- how the quantum circuits are built,
- and how the regression experiments are performed.

Unlike the QSVC notebook, this notebook predicts **continuous stacking fault energy (SFE)** values rather than binary classes. Consequently, one important new hyperparameter—the **epsilon (ε) parameter**—is introduced for Support Vector Regression.

---

# Output Directory

```python
root_folder = "QSVR"
```

This variable specifies the main directory where all outputs generated by the notebook will be stored.

Typical outputs include:

- prediction CSV files,
- result tables,
- plots,
- logs,
- experiment summaries.

For example:

```text
QSVR/

├── result/

├── logs/

├── figures/

└── predictions/
```

Using a dedicated output directory keeps the project organized and separates the regression experiments from the QSVC classification results.

---

# Global Variables

Everything defined in this section acts as a **global variable**.

This means these values can be accessed from anywhere in the notebook without redefining them.

For example,

```python
NUM_FEATURES
```

can be used inside any function that constructs the quantum model.

Global variables make the notebook cleaner and ensure that every experiment uses consistent settings.

---

# Reproducibility

```python
np.random.seed(42)
```

Machine learning algorithms often involve randomness.

For example:

- train-test splits,
- randomized cross-validation,
- random initialization.

Without fixing the random number generator,

each execution of the notebook could produce slightly different results.

Setting

```python
np.random.seed(42)
```

tells NumPy:

> "Whenever random numbers are required, begin from exactly the same starting point."

As a result,

```text
Seed = 42

↓

Same Random Numbers

↓

Same Train/Test Splits

↓

Same Experimental Results
```

This is essential for **scientific reproducibility**.

Another researcher running the notebook should obtain the same results.

---

# Number of Input Features

```python
NUM_FEATURES = 3
```

The model uses three input variables.

These are:

- Electronegativity
- Bulk Modulus
- Atomic Volume

These three material properties become the input vector for every alloy.

---

# Number of Qubits

```python
NUM_QUBITS = NUM_FEATURES
```

This connects the **classical dataset** to the **quantum circuit**.

The ZZFeatureMap encodes one classical feature into one qubit.

Therefore,

```text
Feature 1

↓

Qubit 1
```

```text
Feature 2

↓

Qubit 2
```

```text
Feature 3

↓

Qubit 3
```

Thus,

```text
3 Features

↓

3 Qubits
```

If the dataset contained five features,

the feature map would require five qubits.

---

# Number of Targets

```python
NUM_TARGETS = 1
```

The notebook predicts a single output variable.

```text
Input

Electronegativity

Bulk Modulus

Volume

↓

Output

Stacking Fault Energy (SFE)
```

Unlike QSVC,

this target remains a **continuous numerical value**.

No conversion into classes is performed.

---

# Feature Map Repetitions

```python
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
```

A feature map converts classical numerical data into a quantum state.

The parameter

```text
reps
```

controls how many times the encoding circuit is repeated.

The notebook does not assume one value is optimal.

Instead,

it evaluates five different quantum circuit depths.

```text
Reps = 1

↓

Reps = 2

↓

Reps = 3

↓

Reps = 4

↓

Reps = 5
```

This is a **hyperparameter sweep**.

Every value produces a different quantum feature map.

The experiments later determine which repetition count gives the best regression performance.

---

# Regularization Parameter

```python
REGU_PARA_LIST = [0.1, 1, 10, 100]
```

This list contains the candidate values for the Support Vector Regression regularization parameter,

usually called

```text
C
```

The parameter controls the trade-off between:

- fitting the training data closely,
- and keeping the regression function simple.

Conceptually:

```text
Small C

↓

More Regularization

↓

Simpler Model

↓

Greater Tolerance to Errors
```

```text
Large C

↓

Less Regularization

↓

More Flexible Model

↓

Fits Training Data More Closely
```

Since the best value is unknown,

the notebook tests all four possibilities.

---

# Entanglement Topology

```python
ENTANGLEMENT_LIST = [
    "linear",
    "full",
    "circular"
]
```

This parameter determines how qubits interact inside the `ZZFeatureMap`.

Different interaction patterns create different quantum feature spaces.

---

## Linear

```text
Q0 ─── Q1 ─── Q2
```

Each qubit interacts only with its nearest neighbour.

This produces the simplest quantum circuit.

---

## Full

```text
Q0 ───── Q1
│ \     /
│  \   /
│   \ /
Q2 ────
```

Every qubit interacts with every other qubit.

This creates the richest feature interactions but also the most complex circuit.

---

## Circular

```text
Q0 ─── Q1
│       │
└── Q2 ─┘
```

The first and last qubits are also connected,

forming a ring.

This provides an intermediate level of connectivity.

---

Different entanglement structures produce different quantum kernels.

The notebook later evaluates which topology gives the most accurate regression model.

---

# Repeated Cross-Validation

```python
N_REPEATS = 10
```

The notebook repeats the complete cross-validation experiment ten times.

Why?

Different train-test splits may produce different performance values,

especially for small datasets.

Instead of trusting a single split,

the notebook performs

```text
Experiment

↓

Repeat

↓

Repeat

↓

Repeat

↓

10 Times
```

This produces a much more reliable estimate of model performance.

---

# What Is New Compared to QSVC?

This is the first major difference between the QSVC and QSVR notebooks.

The QSVC notebook contained

```python
CLASSIFIER_THRESHOLD
```

because it converted the stacking fault energy into two categories.

```text
Low SFE

or

High SFE
```

Classification asks:

> **Which class does this alloy belong to?**

Only two possible answers exist.

---

The QSVR notebook does **not** perform this conversion.

Instead,

it predicts the actual numerical stacking fault energy.

Example:

```text
Actual

23.81 mJ/m²

Prediction

24.06 mJ/m²
```

The prediction can take **any continuous value**.

---

# The New Hyperparameter: ε (Epsilon)

Support Vector Regression introduces a new hyperparameter:

```text
ε (epsilon)
```

This parameter does not exist in QSVC.

It defines an **ε-insensitive loss function**, one of the central ideas of Support Vector Regression.

Instead of penalizing every prediction error,

QSVR ignores small errors that fall within an acceptable tolerance.

Suppose

```text
ε = 0.1
```

If the prediction error is

```text
0.05
```

then

```text
0.05 < 0.1

↓

No Penalty
```

The prediction is considered sufficiently accurate.

However,

if the prediction error is

```text
0.35
```

then

```text
0.35 > 0.1

↓

Penalty Applied
```

Only errors larger than ε influence the optimization.

---

# Why Test Multiple ε Values?

The notebook evaluates different epsilon values,

for example:

```text
ε = 0.1
```

and

```text
ε = 0.01
```

because the ideal tolerance is unknown.

A larger epsilon means:

```text
Larger Tolerance

↓

Ignore More Small Errors

↓

Simpler Regression Function
```

A smaller epsilon means:

```text
Smaller Tolerance

↓

Penalize More Errors

↓

Stricter Regression Model
```

The experiments determine which value produces the most accurate predictions for the magnesium alloy dataset.

---

# QSVC vs QSVR

| QSVC | QSVR |
|------|------|
| Classification | Regression |
| Predicts classes | Predicts continuous values |
| Uses `CLASSIFIER_THRESHOLD` | Uses `ε (epsilon)` |
| Output: 0 or 1 | Output: Numerical SFE |
| Decision boundary | Regression function |

Although both algorithms use the same quantum feature map and fidelity kernel,

their learning objectives are fundamentally different.

---

# Research Insight

This configuration cell illustrates how the same quantum kernel framework can support two different machine learning tasks. The quantum components—such as the `ZZFeatureMap`, qubit count, and entanglement topology—remain unchanged between QSVC and QSVR. The primary difference lies in the learning objective. QSVC seeks a decision boundary that separates classes, while QSVR learns a continuous regression function. The introduction of the **ε-insensitive loss** is what transforms the Support Vector Machine from a classifier into a regressor, allowing it to ignore small prediction errors and focus on more significant deviations.

---

# Key Takeaways

- This cell defines the global configuration for the entire QSVR experiment.
- `np.random.seed(42)` ensures that every run is reproducible.
- Three material properties are encoded into **three qubits** using the `ZZFeatureMap`.
- The notebook performs a hyperparameter search over:
  - **Feature map repetitions (`reps`)**
  - **Regularization parameter (`C`)**
  - **Entanglement topology**
- Unlike QSVC, **no classification threshold is used** because the target remains a continuous stacking fault energy value.
- QSVR introduces the **ε (epsilon) parameter**, which defines an **ε-insensitive loss** by ignoring prediction errors smaller than the chosen tolerance.
- Testing multiple ε values allows the researchers to determine the tolerance that provides the best regression performance for the magnesium alloy dataset.



# Preparing One Cross-Validation Fold (`prepare_dataset_k_fold`)

This function prepares **one complete train-test split** for the Quantum Support Vector Regressor (QSVR).

It does **not** build or train the quantum model.

Instead, it performs all the preprocessing required before the QSVR can begin learning.

Its responsibilities include:

- splitting the dataset into training and testing sets,
- separating element names from numerical features,
- scaling the numerical features,
- and returning everything needed for model training and evaluation.

Although this function is almost identical to the one used in the QSVC notebook, there is one important conceptual difference:

> In **QSVC**, `y` contains **class labels** (0 or 1), whereas in **QSVR**, `y` contains **continuous stacking fault energy (SFE) values**.

---

# Function Arguments

```python
prepare_dataset_k_fold(
    X,
    y,
    train_indices,
    test_indices
)
```

The function receives four inputs.

---

## `X` — Input Feature Matrix

`X` contains all input features before any preprocessing.

Example:

| Element | Electronegativity | Bulk Modulus | Volume |
|----------|------------------:|-------------:|-------:|
| Mg-Al | 1.42 | 37 | 14.2 |
| Mg-Zn | 1.65 | 41 | 13.8 |
| Mg-Y | 1.22 | 28 | 18.1 |

Notice that the first column contains the alloy names.

The remaining columns are numerical features used for machine learning.

---

## `y` — Target Values

Unlike QSVC,

the target values remain continuous.

Example:

```text
23.4

18.7

27.1

15.9
```

These values represent the **actual stacking fault energy (SFE)**.

The QSVR attempts to predict these numbers directly.

There is **no conversion into classes**.

---

## `train_indices`

Repeated K-Fold determines which rows belong to the training set.

Example:

```text
Training Samples

0

2

3

5

6

8
```

Internally,

only the row numbers are stored.

```text
[0, 2, 3, 5, 6, 8]
```

These are called the **training indices**.

---

## `test_indices`

Similarly,

the testing samples are stored as indices.

Example:

```text
Testing Samples

1

4

7
```

Internally,

```text
[1, 4, 7]
```

Only these rows will be used to evaluate the model.

---

# Creating the Train-Test Split

```python
X_train_raw, X_test_raw = X[train_indices], X[test_indices]

y_train, y_test = y[train_indices], y[test_indices]
```

This separates the dataset into:

- training features,
- testing features,
- training targets,
- testing targets.

The result is:

```text
Entire Dataset

        │

        ▼

Training Data

+

Testing Data
```

No learning has occurred yet.

The data have simply been divided.

---

# Separating Alloy Names

```python
element_test = X_test_raw[:, 0]

element_train = X_train_raw[:, 0]
```

The first column contains alloy names.

Example:

```text
Mg-Al

Mg-Zn

Mg-Y
```

These names are useful for reporting results,

but they cannot be used as numerical inputs.

Therefore,

they are stored separately.

---

# Removing the Element Column

```python
X_train = X_train_raw[:, 1:]

X_test = X_test_raw[:, 1:]
```

The notebook removes the first column,

leaving only numerical features.

Before:

| Element | EN | Bulk Modulus | Volume |
|----------|---:|-------------:|-------:|
| Mg-Al | 1.42 | 37 | 14.2 |

After:

| EN | Bulk Modulus | Volume |
|---:|-------------:|-------:|
|1.42|37|14.2|

Only these numerical values are passed to the quantum feature map.

---

# Combining the Features

```python
full_X = np.vstack([X_train, X_test])
```

`np.vstack()` means **vertical stack**.

Example:

Training:

```text
1   2   3

4   5   6
```

Testing:

```text
7   8   9
```

After stacking:

```text
1   2   3

4   5   6

7   8   9
```

The training and testing features are temporarily combined into one matrix.

---

# Creating the Scaler

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
```

This creates a **MinMaxScaler** object.

At this stage,

nothing has been transformed.

Think of it like purchasing a ruler.

The ruler exists,

but no measurements have been taken yet.

---

Suppose the raw feature ranges are:

| Feature | Typical Range |
|---------|---------------|
| Electronegativity | 1–4 |
| Bulk Modulus | 20–250 |
| Volume | 10–30 |

Notice that

Bulk Modulus

has much larger numerical values than

Electronegativity.

Without scaling,

large-valued features may dominate the calculations.

Scaling places every feature on the same numerical scale.

---

# Learning the Scaling

```python
scaler.fit(full_X)
```

The `fit()` function calculates:

- minimum value,
- maximum value,

for every feature.

These values are stored internally.

Important:

`fit()` **does not change the data**.

It only learns the transformation.

---

# Scaling the Data

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Now the learned transformation is applied.

Both the training and testing features are scaled using the **same scaler**.

This ensures that

```text
Training Features

↓

Same Feature Space

↓

Testing Features
```

The model therefore receives consistent inputs.

---

# Why Scale the Features?

Quantum feature maps encode classical data into quantum rotations.

If one feature ranges between

```text
1–4
```

while another ranges between

```text
20–250
```

the larger feature could dominate the quantum encoding.

Scaling prevents this imbalance,

allowing all material properties to contribute more equally.

---

# A Small but Important Research Decision

Notice that the scaler is fitted using

```python
full_X
```

which combines

- training data,
- testing data.

This is exactly how the published notebook is written,

so it should be reproduced faithfully.

However,

from a modern machine learning perspective,

many researchers would instead write

```python
scaler.fit(X_train)
```

and then transform

```python
X_train

X_test
```

using only statistics learned from the training data.

This avoids **data leakage** because information from the testing samples is never used during preprocessing.

Therefore,

there are two perspectives:

### To reproduce the published work

```text
Fit on

Training + Testing
```

This exactly matches the original implementation.

---

### To improve the methodology

```text
Fit on

Training Only
```

This follows current best practices in machine learning.

Recognizing these subtle methodological choices is an important step in moving from **code reproduction** to **critical evaluation** and eventually **research innovation**.

---

# Complete Workflow

```text
Input Features (X)
        │
        ▼
Split Into Training and Testing Sets
        │
        ▼
Separate Alloy Names
        │
        ▼
Remove Element Column
        │
        ▼
Combine Numerical Features
        │
        ▼
Create MinMaxScaler
        │
        ▼
Learn Scaling Parameters
        │
        ▼
Scale Training Features
        │
        ▼
Scale Testing Features
        │
        ▼
Return Processed Data
```

---

# Research Insight

Although this function contains no quantum computations, it is a crucial part of the experimental pipeline. Quantum machine learning models are highly sensitive to how classical data are prepared before encoding into quantum states. Proper preprocessing ensures that the numerical features are presented in a consistent range, enabling the `ZZFeatureMap` to encode them effectively. Furthermore, understanding details such as where the scaler is fitted helps distinguish **reproducing an existing implementation** from **designing an improved methodology**, a key skill in computational research.

---

# Key Takeaways

- This function prepares **one train-test split** for a cross-validation fold.
- `X` contains the three material features, while `y` contains the **continuous stacking fault energy values**.
- The element names are separated because they are useful for reporting but cannot be used as numerical inputs.
- Only the numerical features are retained for machine learning.
- `MinMaxScaler` scales all features into the range **[-1, 1]**, ensuring consistent input for the quantum feature map.
- The published notebook fits the scaler using both training and testing data to reproduce the original implementation exactly.
- A common methodological improvement is to fit the scaler **only on the training data** to prevent data leakage during cross-validation.



# Building the Quantum Support Vector Regressor (QSVR)

This function constructs a **Quantum Support Vector Regressor (QSVR)** using a given set of hyperparameters.

Like the corresponding QSVC function, **it only builds the model**.

It does **not** perform any training or prediction.

Its purpose is to assemble all the components required for a Quantum Support Vector Regression model:

- Quantum Feature Map
- Quantum Kernel
- Support Vector Regressor

and return the completed model.

---

# Function Arguments

```python
(
    feature_dimension,
    C,
    epsilon,
    reps,
    entangle
)
```

The function receives five hyperparameters.

Each one controls a different aspect of the regression model.

---

# `feature_dimension`

Earlier we defined

```python
NUM_FEATURES = 3
```

Therefore,

```python
feature_dimension = 3
```

The dataset contains three input variables:

- Electronegativity
- Bulk Modulus
- Atomic Volume

The quantum feature map therefore creates **three encoded quantum inputs**, one for each feature.

```text
Feature 1

↓

Qubit 1
```

```text
Feature 2

↓

Qubit 2
```

```text
Feature 3

↓

Qubit 3
```

---

# `C` — Regularization Parameter

The parameter

```text
C
```

comes from classical Support Vector Machines.

It controls the balance between

- fitting the training data closely,
- and keeping the regression function simple.

A small value

```text
C = 0.1
```

means

```text
More Regularization

↓

Simpler Model

↓

More Tolerant of Errors
```

A larger value

```text
C = 100
```

means

```text
Less Regularization

↓

More Flexible Model

↓

Fits Training Data More Closely
```

Since the optimal value is unknown,

the notebook later tests

```text
0.1

1

10

100
```

This process is called a **hyperparameter sweep**.

---

# `epsilon (ε)` — The New Regression Hyperparameter

Unlike QSVC,

QSVR introduces an additional parameter:

```text
epsilon (ε)
```

The notebook comments describe it as:

> **The epsilon-tube within which prediction errors are ignored during training.**

This is one of the defining ideas of **Support Vector Regression**.

---

## The ε-Insensitive Loss

Suppose

```text
Actual SFE = 24.50
```

The model predicts

```text
24.56
```

The prediction error is

```text
0.06
```

If

```text
ε = 0.10
```

then

```text
0.06 < 0.10
```

The prediction lies **inside the epsilon tube**.

No penalty is applied.

---

Now suppose the prediction is

```text
25.10
```

The error becomes

```text
0.60
```

Since

```text
0.60 > 0.10
```

the prediction lies **outside the epsilon tube**.

The model is penalized for this larger error.

---

Graphically,

```text
          Penalty

              ▲
              │
--------------│-----------------
              │
      ε Tube  │
<-------------0------------->
              │
       No Penalty
```

Small prediction errors inside the tube are ignored.

Only larger errors influence the optimization.

---

# Why Is ε Important?

The epsilon parameter determines how "forgiving" the regression model should be.

A larger epsilon means

```text
Ignore More Small Errors

↓

Simpler Regression Function

↓

Less Sensitive
```

A smaller epsilon means

```text
Ignore Fewer Errors

↓

Stricter Regression Function

↓

More Sensitive
```

The notebook tests multiple epsilon values because it is not known beforehand which tolerance produces the best predictions.

---

# `reps`

The parameter

```text
reps
```

controls the number of times the quantum feature map is repeated.

Earlier,

we defined

```python
FEATURE_MAP_REPS_LIST = [1,2,3,4,5]
```

Each value creates a quantum circuit with a different depth.

Larger repetition counts generally produce more expressive quantum feature maps,

but also increase circuit complexity.

---

# `entangle`

This parameter controls how qubits interact.

Possible values are

```text
linear

full

circular
```

Different interaction patterns generate different quantum feature spaces.

The notebook later evaluates all three topologies.

---

# Creating the Quantum Feature Map

```python
feature_map = ZZFeatureMap(
    feature_dimension=feature_dimension,
    reps=reps,
    entanglement=entangle,
    insert_barriers=True
)
```

The feature map converts classical numerical data into a quantum circuit.

The paper uses the **ZZFeatureMap** because it introduces entanglement between qubits.

Instead of encoding each feature independently,

the quantum circuit can also represent relationships between features.

---

### Why ZZFeatureMap?

Suppose two alloys have

```text
High Electronegativity

and

High Bulk Modulus
```

The ZZFeatureMap allows these properties to interact through entanglement,

making the encoded quantum state more expressive than simply encoding each feature separately.

This richer representation can improve the ability of the quantum kernel to distinguish different materials.

---

# `insert_barriers=True`

A **barrier** is a visual separator inside a quantum circuit diagram.

It does **not** affect the mathematics or computation.

Its only purpose is to make the circuit easier to read.

---

# Building the Quantum Kernel

```python
kernel = FidelityQuantumKernel(
    feature_map=feature_map
)
```

This creates the **quantum kernel**.

A kernel answers one fundamental question:

> **How similar are two samples?**

---

Suppose we compare two alloys.

```text
Mg-Al

Mg-Zn
```

The kernel

- encodes both alloys into quantum states,
- compares those quantum states,
- computes their **quantum state fidelity**.

If the fidelity is

```text
High
```

the alloys are considered similar.

If the fidelity is

```text
Low
```

they are considered different.

Unlike classical SVMs,

which use mathematical formulas such as

- Linear Kernel
- Polynomial Kernel
- Radial Basis Function (RBF)

QSVR measures similarity using **quantum mechanics**.

---

# Constructing the Quantum Support Vector Regressor

```python
qsvr = QSVR(
    C=C,
    epsilon=epsilon,
    quantum_kernel=kernel
)
```

This is where all components come together.

The regressor now knows:

- which quantum feature map to use,
- which quantum kernel to use,
- which regularization parameter (`C`) to use,
- which epsilon (`ε`) defines the insensitive loss.

Notice that the learning algorithm is still based on the familiar **Support Vector Regression** framework.

The "quantum" aspect comes from the kernel computation.

Instead of computing similarities with a classical mathematical formula,

the algorithm computes them using **quantum state fidelities**.

---

# Returning the Model

```python
return qsvr
```

Finally,

the completed model is returned.

At this point,

nothing has been learned.

The returned model is simply an **empty Quantum Support Vector Regressor** ready for training.

Training happens later using

```python
qsvr.fit(...)
```

---

# Overall Workflow

```text
Hyperparameters

(C, ε, reps, entanglement)

        │

        ▼

Create ZZFeatureMap

        │

        ▼

Build Fidelity Quantum Kernel

        │

        ▼

Construct QSVR

        │

        ▼

Return Untrained Model
```

---

# QSVC vs QSVR Construction

| QSVC | QSVR |
|------|------|
| Uses `QSVC` | Uses `QSVR` |
| Predicts classes | Predicts continuous values |
| No epsilon parameter | Includes `epsilon (ε)` |
| Decision boundary | Regression function |
| Quantum kernel | Quantum kernel |

Everything else—the quantum feature map, fidelity kernel, and hyperparameter search—is nearly identical.

The primary difference is that **QSVR introduces the ε-insensitive loss**, allowing it to perform regression instead of classification.

---

# Research Insight

This function demonstrates how quantum machine learning extends classical Support Vector Regression rather than replacing it. The quantum advantage lies **not in a new optimization algorithm**, but in the way sample similarity is computed. Classical SVR uses predefined mathematical kernels, while QSVR replaces them with a **quantum kernel** based on state fidelities. The addition of the **epsilon (ε) parameter** further distinguishes regression from classification by allowing the model to ignore small prediction errors, producing a smoother and more robust regression function.

---

# Key Takeaways

- This function **constructs** a Quantum Support Vector Regressor (QSVR) but does **not train** it.
- The model is built from three main components:
  - **ZZFeatureMap**
  - **FidelityQuantumKernel**
  - **QSVR**
- `feature_dimension` determines the number of qubits used to encode the classical features.
- `C` controls the trade-off between model simplicity and fitting the training data.
- `epsilon (ε)` defines the **ε-insensitive loss**, allowing small prediction errors to be ignored during training.
- The **ZZFeatureMap** encodes classical alloy properties into entangled quantum states.
- The **FidelityQuantumKernel** measures similarity between alloys using quantum state fidelity instead of a classical kernel function.
- The returned model is **untrained** and becomes a learned regression model only after calling `qsvr.fit()`.



# Training the Quantum Support Vector Regressor (`train_qsvr`)

This function is the point where the model changes from an **empty Quantum Support Vector Regressor** into a **trained regression model**.

The previous function (`reconfig_quantum_kernel_qsvr`) only constructed the model.

This function actually teaches it using the training data.

After training, it immediately predicts the stacking fault energies for both:

- the training alloys,
- and the unseen testing alloys.

Unlike QSVC, the predictions are **continuous numerical values**, not class labels.

---

# Function Definition

```python
def train_qsvr(qsvr, X_train, y_train, X_test):
```

## Inputs

- QSVR Model
- Training Features
- Training Target Values
- Testing Features

## Outputs

- Predictions on the Training Set
- Predictions on the Testing Set

Notice that the function **does not return the trained model itself**.

Instead, it returns the numerical predictions produced by the trained model.

---

# `qsvr`

This is the Quantum Support Vector Regressor created earlier.

At this stage, it already knows:

- which **ZZFeatureMap** to use,
- which **Quantum Kernel** to use,
- which **regularization parameter (`C`)** to use,
- which **epsilon (`ε`)** to use.

However,

it has not yet seen any alloy data.

It is simply an empty model waiting to learn.

---

# `X_train`

These are the numerical features used for training.

Example:

| Electronegativity | Bulk Modulus | Volume |
|------------------:|-------------:|-------:|
| -0.30 | 0.65 | -0.10 |
| 0.25 | -0.18 | 0.41 |

Notice that every feature has already been scaled into the range

```text
[-1, 1]
```

These scaled values are what the quantum feature map will encode into quantum states.

---

# `y_train`

Unlike QSVC,

the target values are **continuous stacking fault energies**.

Example:

```text
18.72

23.45

15.91

26.38
```

These are the correct values that the regression model tries to learn.

The objective is **not** to assign a class,

but to predict the numerical stacking fault energy as accurately as possible.

---

# `X_test`

These are the unseen alloys.

The model has never encountered these samples during training.

They are used only to evaluate how well the learned regression function generalizes to new materials.

---

# Training the Model

```python
qsvr.fit(X_train, np.concatenate(y_train))
```

The `.fit()` function is where learning actually happens.

Before this line,

the model knows only its hyperparameters.

After this line,

it has learned a regression function from the training data.

---

## Why `np.concatenate(y_train)`?

The target values are often stored as a column vector.

Example:

```text
[[18.72]

 [23.45]

 [15.91]]
```

However,

QSVR expects a one-dimensional array.

`np.concatenate()` converts the target values into the required format.

Before:

```text
[[18.72]

 [23.45]

 [15.91]]
```

After:

```text
[18.72, 23.45, 15.91]
```

This is simply a formatting step.

It does **not** change the values themselves.

---

# What Happens Inside `.fit()`?

Although the code contains only one line,

a great deal of computation occurs internally.

The workflow is approximately:

```text
Training Features

        │

        ▼

Encode Every Alloy Using ZZFeatureMap

        │

        ▼

Generate Quantum States

        │

        ▼

Compute Quantum Kernel Matrix

(using Fidelity)

        │

        ▼

Solve the Support Vector Regression Optimization Problem

        │

        ▼

Construct the Regression Function
```

Notice that

the quantum computer (or quantum simulator)

is **not directly predicting the stacking fault energy**.

Instead,

its role is to compute the **quantum kernel matrix**.

The actual optimization is still performed by the classical Support Vector Regression algorithm.

---

# Making Predictions

```python
return qsvr.predict(X_train), qsvr.predict(X_test)
```

After training,

the model can estimate the stacking fault energy of any alloy.

The function performs two predictions.

---

## Training Predictions

```python
qsvr.predict(X_train)
```

These predictions are made on the same data used for learning.

They answer the question:

> **How well did the model fit the training alloys?**

---

## Testing Predictions

```python
qsvr.predict(X_test)
```

These predictions are made on completely unseen alloys.

They answer the more important question:

> **Can the model generalize to new materials?**

This is the primary measure of model performance.

---

# What Does the Function Return?

The function returns two arrays.

Example:

Training predictions:

```text
18.65

23.38

16.01

26.21
```

Testing predictions:

```text
21.84

17.32

24.59
```

These are **predicted stacking fault energies**, not class labels.

Later sections of the notebook compare these predictions with the true values using regression metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

# QSVC vs QSVR Prediction

### QSVC

Output:

```text
0

1

0

1
```

The model predicts classes.

---

### QSVR

Output:

```text
18.73

22.15

16.84

25.91
```

The model predicts continuous numerical values.

This is the fundamental difference between classification and regression.

---

# Complete Workflow

```text
Empty QSVR

        │

        ▼

Receive Training Data

        │

        ▼

Encode Features with ZZFeatureMap

        │

        ▼

Compute Quantum Kernel Matrix

        │

        ▼

Solve Support Vector Regression

        │

        ▼

Learn Regression Function

        │

        ▼

Predict Training SFE

        │

        ▼

Predict Testing SFE

        │

        ▼

Return Predictions
```

---

# Research Insight

Although this function appears simple, it encapsulates the entire Quantum Support Vector Regression workflow. Calling `.fit()` does far more than fitting a classical regression model. First, the `ZZFeatureMap` transforms each alloy's classical properties into quantum states. The `FidelityQuantumKernel` then measures the similarity between these quantum states to construct a quantum kernel matrix. Finally, a **classical Support Vector Regression optimizer** uses this kernel matrix to learn a regression function. Thus, the quantum computation enhances the representation of the data, while the optimization itself remains classical.

---

# Key Takeaways

- This function converts an **untrained QSVR** into a **trained regression model**.
- `X_train` contains the scaled material features, while `y_train` contains the **continuous stacking fault energy values**.
- `np.concatenate(y_train)` reshapes the target array into the format expected by QSVR.
- During `.fit()`, Qiskit:
  - encodes the features using the **ZZFeatureMap**,
  - computes the **quantum kernel matrix** using state fidelities,
  - and passes that kernel matrix to a classical Support Vector Regression optimizer.
- After training, the function predicts stacking fault energies for both the training and testing datasets.
- Unlike QSVC, the outputs are **continuous numerical predictions**, making this a regression task rather than a classification task.



# Reading Command-Line Arguments (`get_arguments`)

This function is **not part of the Quantum Support Vector Regressor (QSVR) algorithm**.

Instead, it is an example of **research software engineering**.

Its purpose is to make the program easier to run from the command line by allowing the user to specify different hyperparameters without editing the source code.

In other words:

- the **QSVR algorithm remains exactly the same**,
- only the **way hyperparameters are supplied** changes.

---

# Purpose of the Function

```python
def get_arguments(argvs):
```

The purpose of this function is to read the options provided when the program is executed from a terminal.

Suppose that instead of using a Jupyter Notebook, the project is saved as a Python script named

```text
QSVR.py
```

The script could then be executed like this:

```bash
python QSVR.py -e linear -f 3 -r 10 -p 0.1
```

This command means:

```text
Run QSVR.py

↓

Entanglement = linear

↓

Feature Map Repetitions = 3

↓

Regularization Parameter (C) = 10

↓

Epsilon = 0.1
```

Without this function,

Python would have no idea what

```text
-e

-f

-r

-p
```

represent.

---

# `argvs` — Argument Vector

The parameter

```python
argvs
```

stands for **argument vector**.

It simply contains everything typed after the program name.

For example,

```bash
python QSVR.py -e linear -f 3 -r 10 -p 0.1
```

internally becomes something similar to

```text
[
"-e",
"linear",
"-f",
"3",
"-r",
"10",
"-p",
"0.1"
]
```

The job of this function is to interpret this list.

---

# Creating Empty Variables

At the beginning of the function,

empty variables are created.

```python
_entangle = ""

_feature_map_reps = ""

_regu_para = ""

_epsilon = ""
```

These variables act as **containers** waiting to be filled.

Later,

when command-line arguments are read,

their values become

```text
_entangle = "linear"

_feature_map_reps = "3"

_regu_para = "10"

_epsilon = "0.1"
```

---

# Error Handling

The function uses

```python
try
```

and

```python
except
```

blocks.

Their purpose is simply to prevent the program from crashing if the user enters incorrect arguments.

For example,

if someone writes

```bash
python QSVR.py -z something
```

instead of

```bash
python QSVR.py -e linear
```

the program can display a helpful error message instead of terminating unexpectedly.

This improves the usability of the software.

---

# Parsing the Arguments

The key line is

```python
opts, args = getopt.getopt(
    argvs,
    "h:e:f:r:p:",
    [
        "entangle=",
        "feature_map_reps=",
        "_regu_para=",
        "epsilon="
    ]
)
```

The `getopt` module was imported at the beginning of the notebook.

Its job is to separate the command-line arguments into meaningful options.

---

# Understanding the Flags

The string

```text
"h:e:f:r:p:"
```

defines the available **short command-line flags**.

| Flag | Meaning |
|------|---------|
| `-h` | Help |
| `-e` | Entanglement topology |
| `-f` | Feature map repetitions |
| `-r` | Regularization parameter (`C`) |
| `-p` | Epsilon (`ε`) |

Notice the new addition compared to QSVC:

```text
-p

↓

Epsilon
```

This parameter exists because QSVR performs **regression**, where the ε-insensitive loss is an essential part of the algorithm.

---

# Why Is Epsilon Passed from the Command Line?

Earlier,

we learned that epsilon controls the width of the **ε-insensitive tube**.

Different values produce different regression behaviour.

For example,

```bash
python QSVR.py -p 0.01
```

creates a stricter regression model,

whereas

```bash
python QSVR.py -p 0.10
```

allows a wider tolerance for prediction errors.

Instead of editing the Python file each time,

the researcher simply changes the command-line option.

---

# Does This Change the Quantum Algorithm?

No.

From a **Quantum Machine Learning** perspective,

this function changes absolutely nothing.

Whether the hyperparameters are

- typed directly into the notebook,

or

- supplied through command-line arguments,

the mathematical model remains identical.

The same:

- ZZFeatureMap,
- Fidelity Quantum Kernel,
- QSVR optimization,

are used.

Only the way the values are supplied changes.

---

# Why Researchers Use This

Imagine running hundreds of experiments.

Instead of editing the notebook repeatedly,

a researcher can launch many experiments automatically.

Example:

```bash
python QSVR.py -e linear -f 1 -r 0.1 -p 0.01

python QSVR.py -e linear -f 2 -r 0.1 -p 0.01

python QSVR.py -e full -f 5 -r 100 -p 0.10

python QSVR.py -e circular -f 3 -r 10 -p 0.01
```

Every command launches a different experiment.

No source code needs to be modified.

---

# Why Is This Useful on HPC Systems?

Suppose a workstation or High-Performance Computing (HPC) cluster contains hundreds of CPU cores.

Instead of manually running experiments,

a shell script can automatically execute all possible combinations.

Example:

```text
C Values

×

Feature Map Repetitions

×

Entanglement Types

×

Epsilon Values

↓

Hundreds of QSVR Experiments

↓

Automatically
```

This makes large-scale hyperparameter searches practical.

---

# What Does This Tell Us About the Notebook?

The presence of this function suggests something interesting.

It is likely that the notebook was **adapted from a standalone Python script**.

Originally,

the authors probably executed the experiments from a Linux terminal or HPC cluster,

where command-line arguments are much more convenient than editing notebook cells.

The notebook version simply retains this functionality.

---

# Overall Workflow

```text
Run Python Script

        │

        ▼

Read Command-Line Arguments

        │

        ▼

Extract:

• Entanglement

• Feature Map Repetitions

• Regularization (C)

• Epsilon (ε)

        │

        ▼

Pass Hyperparameters

        │

        ▼

Construct QSVR

        │

        ▼

Run Experiment
```

---

# Research Insight

Although this function contains no machine learning or quantum computing, it represents an important aspect of **reproducible computational research**. Modern research projects often require thousands of automated experiments to explore different hyperparameter combinations. By allowing values such as **C**, **feature map repetitions**, **entanglement topology**, and **epsilon** to be supplied through command-line arguments, researchers can use shell scripts or job schedulers to perform large-scale parameter sweeps without modifying the source code. This separation between the algorithm and the experiment configuration is considered good scientific software design.

---

# Key Takeaways

- This function is **not part of the QSVR algorithm**; it is a software engineering utility.
- It allows hyperparameters to be supplied from the command line instead of being hard-coded.
- The supported command-line options include:
  - **Entanglement topology (`-e`)**
  - **Feature map repetitions (`-f`)**
  - **Regularization parameter (`-r`)**
  - **Epsilon (`-p`)**
- `argvs` stores everything typed after the program name and is parsed using Python's `getopt` module.
- `try` and `except` blocks improve robustness by handling invalid arguments gracefully.
- The function does **not** change the mathematics of QSVR; it only changes how experiment settings are provided.
- Command-line arguments make it easy to automate large batches of experiments on Linux workstations and HPC clusters, improving reproducibility and research efficiency.



# Creating Output Directories

This cell is **not related to the Quantum Support Vector Regressor (QSVR) algorithm itself**.

Instead, it is another example of **research software engineering**.

Its purpose is to automatically create the folders needed to store the results generated during the experiments.

The previous function (`get_arguments`) handled the **input** to the program by reading hyperparameters from the command line.

This cell handles the **output** by preparing a place to save the experiment results.

---

# The Code

```python
if not os.path.exists(f"{root_folder}/result"):
    os.makedirs(f"{root_folder}/result")

if not os.path.exists(f"{root_folder}/logs"):
    os.makedirs(f"{root_folder}/logs")
```

This code checks whether two directories already exist:

- `QSVR/result`
- `QSVR/logs`

If they do not exist,

Python creates them automatically.

---

# Understanding `os.path.exists()`

The function

```python
os.path.exists(path)
```

asks a simple question:

> **"Does this folder (or file) already exist?"**

For example,

```python
os.path.exists("QSVR/result")
```

returns

```text
True
```

if the folder is already present,

and

```text
False
```

if it does not exist.

---

# Understanding `os.makedirs()`

If the folder does not exist,

the notebook executes

```python
os.makedirs(path)
```

which creates the required directory.

For example,

```python
os.makedirs("QSVR/result")
```

creates

```text
QSVR/

└── result/
```

Likewise,

```python
os.makedirs("QSVR/logs")
```

creates

```text
QSVR/

└── logs/
```

If the folders already exist,

nothing happens.

---

# Decision Flow

The logic can be visualized as:

```text
Start Program

      │

      ▼

Is "QSVR/result" present?

      │

 ┌────┴────┐

 │         │

Yes       No

 │         │

Skip    Create Folder

 │

 ▼

Is "QSVR/logs" present?

      │

 ┌────┴────┐

 │         │

Yes       No

 │         │

Skip    Create Folder

 │

 ▼

Continue Experiment
```

This ensures that the notebook always has the required folder structure before any files are written.

---

# Why Is This Necessary?

Machine learning experiments generate many different files.

For example,

- prediction CSV files,
- regression metrics,
- plots,
- logs,
- experiment summaries.

Without dedicated folders,

all of these files would accumulate in one location,

making the project difficult to manage.

Instead,

the notebook organizes everything automatically.

Example:

```text
QSVR/

├── result/
│   ├── experiment_01.csv
│   ├── experiment_02.csv
│   └── experiment_03.csv
│
├── logs/
│   ├── training.log
│   └── execution.log
│
└── QSVR.py
```

This makes the project much easier to navigate.

---

# Why Check Before Creating?

Suppose the folders already exist.

If Python tried to create them again,

an error would occur.

Therefore,

the notebook first checks

```python
os.path.exists(...)
```

Only if the answer is

```text
False
```

does it execute

```python
os.makedirs(...)
```

This prevents unnecessary errors and allows the notebook to be run repeatedly.

---

# Does This Affect the Quantum Model?

No.

This cell has **no influence whatsoever** on

- the ZZFeatureMap,
- the Fidelity Quantum Kernel,
- the QSVR algorithm,
- the training process,
- or the regression results.

Whether the folders are created automatically or manually,

the mathematics of the experiment remains identical.

Its purpose is purely organizational.

---

# Why Researchers Use This

Imagine performing hundreds of QSVR experiments.

Each experiment might generate:

- prediction tables,
- error metrics,
- plots,
- log files.

Without an organized directory structure,

the project could quickly become chaotic.

Automatic folder creation ensures that every experiment stores its outputs in the correct location.

This improves:

- organization,
- reproducibility,
- portability.

---

# Why Is This Good Research Practice?

Suppose another researcher downloads the project onto a different computer.

Initially,

none of the folders exist.

Instead of asking the user to manually create

```text
result/

logs/
```

the notebook does it automatically.

The project therefore becomes **portable**.

It can be executed on:

- Windows,
- Linux,
- macOS,
- HPC clusters,

without requiring any manual setup of output directories.

---

# Overall Workflow

```text
Start Program

        │

        ▼

Check "result" Folder

        │

        ▼

Create If Missing

        │

        ▼

Check "logs" Folder

        │

        ▼

Create If Missing

        │

        ▼

Begin QSVR Experiment
```

---

# Research Insight

Although this cell contains no machine learning or quantum computing, it reflects an important principle of **computational research**: experiments should organize themselves. Scientific workflows often produce hundreds or thousands of output files. Automatically creating the required directory structure ensures that results are stored consistently, reduces the chance of user error, and makes the project easy to reproduce on different computers. Good software organization is an essential part of reproducible scientific computing.

---

# Key Takeaways

- This cell is **not part of the QSVR algorithm**; it is a software engineering utility.
- It automatically creates the required output directories if they do not already exist.
- `os.path.exists()` checks whether a folder is already present.
- `os.makedirs()` creates the folder only when it is missing.
- The folders are typically used to store:
  - **Regression result CSV files**
  - **Training and execution logs**
- Automatic directory creation improves:
  - **Organization**
  - **Portability**
  - **Reproducibility**
- This is an example of **good computational research practice**, ensuring that experiments can be run on different systems without manual preparation.



# Loading and Preparing the Dataset

This cell is the point where the notebook loads the experimental alloy dataset into memory.

Unlike the previous cells, which dealt with software organization and experiment setup, this cell introduces the **actual materials science data** that the Quantum Support Vector Regressor (QSVR) will learn from.

Its primary purpose is to:

1. Load the dataset from a CSV file.
2. Separate the input features from the target variable.
3. Convert the data into NumPy arrays.
4. Inspect the dataset dimensions before training begins.

---

# Locating the Dataset

```python
dataset_name = "./Desktop/ABHI/Learning,Reproducing/qml_training-validation-data.csv"
```

This line stores the location of the dataset in a variable called `dataset_name`.

Instead of typing the file path multiple times,

the notebook stores it once and reuses it whenever the dataset needs to be accessed.

---

# Reading the CSV File

The dataset is loaded using

```python
pd.read_csv(dataset_name)
```

This function belongs to the **Pandas** library.

Its job is to read the CSV file and convert it into a **DataFrame**.

Conceptually,

```text
CSV File

↓

Pandas DataFrame
```

A DataFrame is a table-like structure where rows represent observations and columns represent variables.

---

# Displaying the Dataset

The notebook uses

```python
display(df)
```

instead of

```python
print(df)
```

The difference is mainly visual.

`display()` renders the dataset as a properly formatted table, making it much easier to inspect.

For example,

| Element | Electronegativity | Bulk Modulus | Volume | SFE |
|----------|------------------:|-------------:|-------:|----:|
| Mg-Al | 1.42 | 37 | 14.2 | 18.5 |
| Mg-Zn | 1.65 | 41 | 13.8 | 21.3 |
| Mg-Y | 1.22 | 28 | 18.1 | 12.7 |

This tabular format is much easier to read than plain text output.

---

# Separating Inputs and Target

Machine learning requires us to distinguish between:

- **inputs (features)**, and
- **outputs (targets).**

The notebook creates

```python
X
```

for the input features,

and

```python
y
```

for the target variable.

---

# Input Features (`X`)

The following columns are selected:

```python
[
'Element',
'el_neg',
'B/GPa',
'Volume/A^3'
]
```

Notice the use of **double square brackets**:

```python
df[[ ... ]]
```

Double brackets tell Pandas to select **multiple columns**.

These columns become the model inputs.

The features are:

| Feature | Physical Meaning |
|----------|------------------|
| Element | Alloy composition (used only as an identifier) |
| Electronegativity (`el_neg`) | Average electronegativity of the alloy |
| Bulk Modulus (`B/GPa`) | Resistance to compression |
| Atomic Volume (`Volume/A³`) | Average atomic volume |

These properties describe each alloy.

---

# Target Variable (`y`)

The notebook selects

```python
'SFE/mJm^-3'
```

using **single brackets**.

```python
df['SFE/mJm^-3']
```

A single bracket selects **one column only**.

This column contains the property that the model must predict.

In this notebook,

that property is

```text
Stacking Fault Energy (SFE)
```

Unlike the QSVC notebook,

the values remain **continuous** because this is a **regression problem**.

For example,

```text
18.4

22.7

15.9

27.1
```

The model predicts numerical values rather than discrete classes.

---

# Converting to NumPy Arrays

After selecting the required columns,

the notebook calls

```python
.values
```

This converts the Pandas DataFrame into a **NumPy array**.

Conceptually,

```text
Pandas DataFrame

↓

NumPy Array
```

Why is this necessary?

Many machine learning libraries,

including **scikit-learn** and **Qiskit Machine Learning**,

operate more efficiently on NumPy arrays than on Pandas DataFrames.

---

# What Does the Model Learn?

After this preprocessing,

the model receives:

```text
Electronegativity

↓

Bulk Modulus

↓

Atomic Volume

↓

Predict

↓

Stacking Fault Energy
```

The first column,

`Element`,

is retained only as an identifier.

It helps us know which alloy is being evaluated,

but it is **not used as a numerical feature during training**.

---

# Inspecting the Dataset Size

The notebook then uses

```python
X.shape
```

The `.shape` attribute reports the dimensions of the dataset.

For example,

```text
(21, 5)
```

means

- **21 rows** (21 alloy samples),
- **5 columns**.

The five columns correspond to:

```text
Element

Electronegativity

Bulk Modulus

Atomic Volume

Target Property
```

Knowing the dataset size is important because it influences later choices such as cross-validation strategy.

Since the dataset contains only **21 samples**,

the authors later employ **Leave-One-Out Cross Validation (LOOCV)** to obtain a more reliable evaluation.

---

# Why Is This Cell Important?

Everything that follows depends on this dataset.

Without loading and organizing the data,

there would be nothing for the QSVR model to learn.

This cell establishes the connection between the **materials science problem** and the **machine learning algorithm**.

---

# Overall Workflow

```text
Locate CSV File

        │

        ▼

Read CSV with Pandas

        │

        ▼

Create DataFrame

        │

        ▼

Select Input Features (X)

        │

        ▼

Select Target Variable (y)

        │

        ▼

Convert to NumPy Arrays

        │

        ▼

Inspect Dataset Shape

        │

        ▼

Ready for Model Training
```

---

# Research Insight

This cell represents a standard **data preparation** step in machine learning workflows. The model itself does not interact directly with CSV files; it requires numerical arrays containing input features and target values. By clearly separating the features (`X`) from the target (`y`) and converting them into NumPy arrays, the notebook prepares the data in a format that both scikit-learn and Qiskit Machine Learning can efficiently process. Although simple, this preprocessing stage is essential for every supervised learning experiment.

---

# Key Takeaways

- This cell loads the alloy dataset from a CSV file into memory.
- `pd.read_csv()` converts the CSV file into a Pandas DataFrame.
- `display(df)` presents the dataset in a readable table format.
- The input feature matrix **`X`** consists of:
  - **Element**
  - **Electronegativity (`el_neg`)**
  - **Bulk Modulus (`B/GPa`)**
  - **Atomic Volume (`Volume/A³`)**
- The target variable **`y`** is the continuous **Stacking Fault Energy (SFE)**.
- `.values` converts Pandas objects into NumPy arrays suitable for machine learning libraries.
- `.shape` reports the size of the dataset (e.g., **21 samples × 5 columns**), providing useful information for designing the training and evaluation strategy.



# Scaling the Target Variable

Unlike the corresponding section in the QSVC notebook, this cell **does not convert the target into classes**.

Instead, it prepares the **continuous target values** for Quantum Support Vector Regression (QSVR) by scaling them into a numerical range that is more suitable for machine learning.

This is an important distinction between **classification** and **regression**.

---

# Why Is This Needed?

The target variable in this notebook is

```text
Stacking Fault Energy (SFE)
```

Unlike QSVC,

where the model predicts

```text
Low SFE

or

High SFE
```

QSVR predicts the **actual numerical value** of the stacking fault energy.

For example,

```text
18.4

22.7

15.9

27.1
```

These values remain continuous throughout the training process.

---

# Reshaping the Target

Before scaling,

the notebook performs

```python
reshape(-1, 1)
```

Suppose the target values originally look like this:

```text
18.4
22.7
15.9
27.1
```

Internally,

NumPy stores this as a **one-dimensional array**.

After

```python
reshape(-1, 1)
```

it becomes

```text
[
 [18.4]
 [22.7]
 [15.9]
 [27.1]
]
```

Notice the difference.

Before:

```text
18.4
22.7
15.9
27.1
```

After:

```text
[
 [18.4]
 [22.7]
 [15.9]
 [27.1]
]
```

The number of rows remains exactly the same.

Only the shape changes.

---

# Why Is Reshaping Necessary?

Many preprocessing functions in **scikit-learn** expect the data to be a **two-dimensional matrix**.

A column vector tells the scaler that

> **each row represents one sample, and the column contains the target values.**

Without reshaping,

the scaler may raise an error because it cannot distinguish between samples and features.

---

# Scaling the Target

After reshaping,

the notebook applies

```python
fit_transform()
```

using a `MinMaxScaler`.

The scaler first learns the minimum and maximum values of the target variable,

and then rescales every value into the range

```text
-1

↓

1
```

For example,

suppose the original SFE values are

```text
12

18

24

30
```

After scaling,

they might become

```text
-1.0

-0.33

0.33

1.0
```

The exact values depend on the minimum and maximum SFE values present in the dataset.

---

# Why Scale the Target?

The input features were already scaled earlier.

Scaling the target provides a similar benefit.

Machine learning algorithms often perform better when both the inputs and outputs lie within comparable numerical ranges.

This helps improve numerical stability during optimization and prevents very large target values from dominating the learning process.

---

# What Is Different from QSVC?

This is where the regression notebook differs significantly from the classification notebook.

In QSVC,

the target values were converted into classes using a threshold.

For example,

```text
SFE < 19

↓

Class 0

SFE ≥ 19

↓

Class 1
```

The model no longer cared about the exact SFE value.

It only learned whether the alloy belonged to the **low-SFE** or **high-SFE** category.

---

In QSVR,

**no classifier threshold is used**.

Instead,

the original continuous values are preserved.

For example,

```text
18.4

22.7

15.9

27.1
```

remain numerical targets,

only scaled to a different range.

This allows the model to predict the **actual stacking fault energy**, not just its category.

---

# Classification vs Regression

The difference can be summarized as:

### QSVC

```text
Original SFE

↓

Apply Threshold

↓

Class 0 or Class 1

↓

Classification
```

---

### QSVR

```text
Original SFE

↓

Scale to [-1, 1]

↓

Continuous Numerical Values

↓

Regression
```

---

# Why Is Scaling Reversed Later?

After the model finishes making predictions,

those predictions are still in the scaled range

```text
-1

↓

1
```

Before reporting the final results,

the notebook applies

```python
inverse_transform()
```

This converts the predictions back into their original physical units:

```text
mJ/m²
```

so that the predicted stacking fault energies can be directly compared with the experimental measurements.

---

# Overall Workflow

```text
Original SFE Values

        │

        ▼

reshape(-1, 1)

        │

        ▼

Column Vector

        │

        ▼

Fit MinMaxScaler

        │

        ▼

Scale to [-1, 1]

        │

        ▼

Train QSVR

        │

        ▼

Inverse Transform Predictions

        │

        ▼

Predicted SFE (Original Units)
```

---

# Research Insight

Unlike classification, regression requires the model to learn the exact numerical relationship between the input features and the target property. Therefore, the stacking fault energy values are **not converted into classes**. Instead, they are scaled to a standardized numerical range, improving the stability of the optimization process while preserving the continuous nature of the prediction. After training, the predictions are transformed back into their original units so that the model's performance can be evaluated using physically meaningful values.

---

# Key Takeaways

- This cell prepares the **continuous target variable** for Quantum Support Vector Regression.
- `reshape(-1, 1)` converts the one-dimensional target array into a column vector required by scikit-learn.
- `fit_transform()` with `MinMaxScaler` scales the target values to the range **[-1, 1]**.
- Scaling improves numerical stability during model training.
- **Unlike QSVC, no classifier threshold is applied.**
- QSVR predicts the **actual stacking fault energy values**, not binary classes.
- After prediction, `inverse_transform()` converts the scaled predictions back to the original physical units for interpretation and evaluation.



# Creating the Cross-Validation Strategy

This cell creates the **cross-validation object** that will later control how the dataset is divided into training and testing sets.

It is important to understand that this line **does not train the Quantum Support Vector Regressor (QSVR)**.

Instead, it simply creates a **plan** that specifies how every experiment will split the data.

---

# The Code

```python
rkf = RepeatedKFold(
    n_splits=X.shape[0] // TEST_SIZE,
    n_repeats=N_REPEATS
)
```

Conceptually, this means:

> **"Here is how we will divide the dataset during every experiment."**

No learning occurs yet.

The QSVR model has not even been created at this stage.

---

# Why Do We Need Cross-Validation?

Suppose we have **100 alloy samples**.

One simple approach would be:

```text
80 Samples → Training

20 Samples → Testing
```

The model is trained once,

tested once,

and the experiment is finished.

```text
Train Once

↓

Test Once

↓

Final Accuracy
```

Although simple,

this approach has an important weakness.

If the test samples happen to be unusually easy or unusually difficult,

the reported performance may not accurately represent the true predictive ability of the model.

This issue becomes much more serious when working with **small datasets**, such as the 21-alloy dataset used in this study.

---

# What Is Cross-Validation?

Instead of evaluating the model using only one train-test split,

cross-validation repeats the experiment using **many different splits**.

Each sample gets an opportunity to appear in both the training set and the testing set.

Conceptually:

```text
Split Dataset

↓

Train Model

↓

Test Model

↓

Repeat with Different Split

↓

Average Performance
```

This provides a much more reliable estimate of how well the model generalizes to unseen data.

---

# What Is `RepeatedKFold`?

The notebook uses

```python
RepeatedKFold
```

from **scikit-learn**.

It performs two tasks:

1. Divide the dataset into **K folds**.
2. Repeat the entire process multiple times using different random splits.

The procedure becomes:

```text
Divide Dataset into K Folds

        │

        ▼

Train on K − 1 Folds

        │

        ▼

Test on Remaining Fold

        │

        ▼

Repeat Until Every Fold Has Been Tested

        │

        ▼

Randomly Shuffle

        │

        ▼

Repeat Entire Process Again
```

This reduces the influence of randomness and provides a more stable estimate of model performance.

---

# Understanding `n_splits`

The notebook specifies

```python
n_splits = X.shape[0] // TEST_SIZE
```

Earlier,

we observed that

```python
X.shape
```

contains

```text
21 samples
```

and

```python
TEST_SIZE = 1
```

Therefore,

```text
21 // 1 = 21
```

which means

```python
n_splits = 21
```

The dataset is therefore divided into **21 folds**.

---

# What Does That Mean?

If there are **21 folds**,

each fold contains

```text
1 test sample
```

while the remaining

```text
20 samples
```

are used for training.

For example,

```text
Fold 1

Training: 20 alloys

Testing: Alloy 1
```

```text
Fold 2

Training: 20 alloys

Testing: Alloy 2
```

```text
Fold 3

Training: 20 alloys

Testing: Alloy 3
```

This continues until every alloy has been used once as the testing sample.

---

# Leave-One-Out Cross Validation (LOOCV)

When

```text
Number of Folds

=

Number of Samples
```

the method is called

## Leave-One-Out Cross Validation (LOOCV)

The workflow is:

```text
21 Alloys

↓

Leave Alloy 1 Out

↓

Train on Remaining 20

↓

Predict Alloy 1

↓

Leave Alloy 2 Out

↓

Train Again

↓

Predict Alloy 2

↓

...

↓

Leave Alloy 21 Out

↓

Train Again

↓

Predict Alloy 21
```

Every alloy is tested exactly once.

---

# Why Use LOOCV?

LOOCV is particularly valuable when datasets are small.

Since almost the entire dataset is used for training during each iteration,

the model has access to as much information as possible while still being evaluated on unseen data.

This often provides a less biased estimate of model performance than a simple train-test split.

The trade-off is computational cost,

because the model must be trained many times.

---

# Understanding `n_repeats`

The notebook also specifies

```python
n_repeats = N_REPEATS
```

Earlier,

we defined

```python
N_REPEATS = 10
```

(or `1` during some experiments).

This means that the entire LOOCV procedure is repeated multiple times.

Conceptually:

```text
LOOCV

↓

Repeat Again

↓

Repeat Again

↓

...

↓

Repeat 10 Times
```

Repeating the experiment reduces the influence of random data ordering and produces more reliable performance statistics.

---

# Printing the Cross-Validation Object

If we execute

```python
print(rkf)
```

the output is similar to

```text
RepeatedKFold(
    n_repeats=1,
    n_splits=21,
    random_state=None
)
```

This output simply summarizes the evaluation strategy.

It confirms that:

- there are **21 folds**, and
- the procedure will be repeated the specified number of times.

---

# Does This Train the QSVR?

No.

At this stage,

the notebook has **not**:

- created a quantum feature map,
- built a fidelity quantum kernel,
- constructed a QSVR model,
- trained the model,
- made predictions.

It has only created the **evaluation strategy**.

The actual training begins later when the notebook loops over the train-test splits.

---

# Overall Workflow

```text
Load Dataset

        │

        ▼

Create RepeatedKFold Object

        │

        ▼

21 Folds (LOOCV)

        │

        ▼

Repeat Entire Process

        │

        ▼

Training/Test Splits Ready

        │

        ▼

QSVR Training Begins Later
```

---

# Research Insight

The authors use **Repeated Leave-One-Out Cross Validation (LOOCV)** because the dataset contains only **21 alloy samples**. With such limited data, a traditional train-test split would leave too few samples for training and produce highly variable results. LOOCV maximizes the amount of training data in every iteration while still testing the model on unseen samples. Repeating the entire procedure multiple times further improves the robustness of the evaluation by reducing the influence of random data ordering.

---

# Key Takeaways

- This cell **does not train the QSVR model**; it only defines the evaluation strategy.
- `RepeatedKFold` repeatedly divides the dataset into training and testing sets.
- `n_splits = 21` means the dataset is divided into **21 folds**, each containing **one test sample**.
- This evaluation strategy is known as **Leave-One-Out Cross Validation (LOOCV)**.
- `n_repeats` repeats the entire LOOCV procedure multiple times to obtain more reliable performance estimates.
- LOOCV is particularly well suited to **small materials science datasets**, where maximizing the amount of training data is important.
- The cross-validation object created here will later provide the train-test splits used during QSVR training and evaluation.



# Creating the Results DataFrame

This cell creates the **DataFrame** that will store the results of every Quantum Support Vector Regression (QSVR) experiment.

Unlike the previous cell, which planned how the dataset would be split, this cell prepares the notebook to **record the outcome** of every experiment.

Think of it as creating a **digital laboratory notebook**.

Every time a QSVR model is trained and evaluated, the notebook writes its results into this table.

---

# Purpose of This Cell

The objective is simple:

> **Create a structured table that records every experiment performed during the hyperparameter search.**

Instead of storing only the final average performance,

the notebook preserves the results from **every individual QSVR model**.

This allows the experiments to be analyzed, verified, and reproduced later.

---

# What Will Be Stored?

Each experiment contributes one new row to the DataFrame.

Typical information includes:

- Regularization Parameter (**C**)
- Feature Map Repetitions (**Reps**)
- Epsilon (**ε**)
- Entanglement Topology
- Test Alloy
- Actual Stacking Fault Energy
- Predicted Stacking Fault Energy
- Performance Metrics

Conceptually,

```text
Train QSVR

      │

      ▼

Predict Test Alloy

      │

      ▼

Store

C

Reps

Epsilon

Entanglement

Test Alloy

Actual Value

Prediction

↓

Repeat

↓

Large Results Table

↓

Statistical Analysis

↓

Paper Figures
```

---

# Why Store Every Experiment?

Imagine training **thousands of QSVR models**.

If only the final average error were stored,

valuable information would be lost.

For example,

suppose one alloy consistently produces large prediction errors.

Without recording every prediction,

it would be impossible to identify that alloy.

By storing every experiment,

the researchers can later investigate questions such as:

- Which hyperparameter combination performs best?
- Which alloy is the hardest to predict?
- Does increasing the number of feature map repetitions improve accuracy?
- Which entanglement topology performs best?
- Does changing ε improve regression performance?

---

# The DataFrame as an Experiment Logbook

The DataFrame functions much like a laboratory notebook.

Instead of writing observations by hand,

the notebook automatically records every experiment.

For example,

| C | Reps | ε | Entanglement | Test Alloy | Actual SFE | Predicted SFE |
|---:|----:|---:|--------------|------------|-----------:|--------------:|
| 0.1 | 1 | 0.01 | Linear | Mg-Al | 18.4 | 18.1 |
| 10 | 3 | 0.10 | Full | Mg-Zn | 22.7 | 22.9 |
| 100 | 5 | 0.01 | Circular | Mg-Y | 15.9 | 15.4 |

Each row represents one complete QSVR experiment.

---

# Why Include Hyperparameters?

Notice that the DataFrame stores not only the predictions,

but also the hyperparameters used to produce them.

For example,

```text
C = 10

↓

Reps = 3

↓

ε = 0.01

↓

Entanglement = Full
```

This makes every prediction traceable.

Months later,

the researchers can identify exactly which model produced each result.

---

# Difference from the QSVC Notebook

The overall idea is identical to the QSVC notebook,

but there is one important difference.

QSVC stored:

```text
Actual Class

↓

Predicted Class
```

because it solved a **classification problem**.

QSVR stores:

```text
Actual Stacking Fault Energy

↓

Predicted Stacking Fault Energy
```

because it solves a **regression problem**.

Therefore,

the stored values are continuous numerical quantities rather than class labels.

---

# Why Is This Important?

Machine learning research rarely ends after training.

Most of the scientific analysis happens **after** all experiments have finished.

Researchers use this DataFrame to:

- compute average performance,
- compare hyperparameters,
- identify difficult alloys,
- generate tables,
- create plots,
- prepare figures for publication.

Without preserving every experiment,

these analyses would not be possible.

---

# Research Workflow

The complete workflow becomes:

```text
Choose Hyperparameters

        │

        ▼

Build QSVR

        │

        ▼

Train Model

        │

        ▼

Predict Test Alloy

        │

        ▼

Store Results

        │

        ▼

Repeat Thousands of Times

        │

        ▼

Large Results DataFrame

        │

        ▼

Statistical Analysis

        │

        ▼

Research Paper Figures
```

---

# Does This Affect the QSVR Algorithm?

No.

This DataFrame has **no influence** on:

- the quantum feature map,
- the fidelity quantum kernel,
- the QSVR optimization,
- or the prediction process.

Its purpose is purely to **record** the outputs generated by the model.

It is a tool for experiment management rather than machine learning.

---

# Research Insight

This DataFrame serves as the **experiment logbook** for the entire study. Instead of keeping only the final average regression performance, the authors preserve every hyperparameter configuration, every prediction, and every measured value. This makes the experiments fully traceable and reproducible. It also enables detailed post-experiment analyses, such as identifying difficult alloys, comparing hyperparameter combinations, and generating the statistical tables and figures presented in the research paper.

---

# Key Takeaways

- This cell creates the **DataFrame** used to store the results of every QSVR experiment.
- Each row corresponds to one trained and evaluated QSVR model.
- The stored information typically includes:
  - **Regularization Parameter (C)**
  - **Feature Map Repetitions (Reps)**
  - **Epsilon (ε)**
  - **Entanglement Topology**
  - **Test Alloy**
  - **Actual Stacking Fault Energy**
  - **Predicted Stacking Fault Energy**
  - **Performance Metrics**
- Unlike QSVC, the DataFrame stores **continuous regression values** rather than binary class labels.
- Recording every experiment improves **traceability**, **reproducibility**, and enables detailed statistical analysis after all experiments have been completed.



# Generating Descriptive Experiment Filenames

This cell is **not related to the Quantum Support Vector Regressor (QSVR) algorithm itself**.

Instead, it is concerned with **experiment management**.

Its purpose is to automatically generate a descriptive filename that uniquely identifies each experiment.

This is a common practice in computational research, where hundreds or even thousands of experiments may be performed.

---

# Aim of This Cell

The goal is simple:

> **Generate a filename that tells you exactly which hyperparameters produced the results stored in that file.**

Instead of creating files with generic names such as

```text
result.csv
```

or

```text
output.csv
```

the notebook creates filenames that contain the experimental settings.

This allows researchers to identify an experiment simply by reading its filename.

---

# Example Filename

```text
QSVR/result/FMR_1_R_0.1_E_['linear', 'full', 'circular']_EP_0.01_8_19_25_1.csv
```

At first glance, this may look complicated.

However, every part of the filename carries useful information.

---

# Breaking Down the Filename

## Folder

```text
QSVR/result/
```

This tells us where the file is stored.

The file belongs to the **result** directory created earlier in the notebook.

---

## Feature Map Repetitions

```text
FMR_1
```

means

```text
Feature Map Repetitions = 1
```

The quantum feature map was repeated once while encoding the classical data into a quantum circuit.

---

## Regularization Parameter

```text
R_0.1
```

means

```text
C = 0.1
```

The QSVR model used a regularization parameter of **0.1**.

---

## Entanglement Topology

```text
E_['linear', 'full', 'circular']
```

indicates the entanglement configuration associated with the experiment.

In practice, individual experiment files often contain only one topology, such as

```text
E_linear
```

or

```text
E_full
```

or

```text
E_circular
```

If the complete list appears in the filename, it usually reflects how the notebook generated filenames or stored metadata for a batch of experiments.

---

## Epsilon

One important addition compared with the QSVC notebook is

```text
EP_0.01
```

This means

```text
Epsilon (ε) = 0.01
```

This parameter is unique to **Support Vector Regression**.

It specifies the width of the **ε-insensitive tube**, which determines how much prediction error is tolerated before a penalty is applied during training.

Including ε in the filename makes it immediately clear which regression model produced the results.

---

## Remaining Numbers

```text
8_19_25_1
```

These values are additional experiment identifiers.

Their exact meaning depends on the notebook implementation.

They may represent information such as:

- Random seed
- Dataset identifier
- Experiment number
- Fold number
- Timestamp

Their precise purpose can only be confirmed by examining the code that constructs the filename.

---

# Why Is This Useful?

Imagine running hundreds of experiments.

Without descriptive filenames,

you might end up with

```text
result.csv

result_new.csv

result_final.csv

result_final2.csv

result_latest.csv
```

After a few weeks,

it becomes almost impossible to remember which experiment produced which file.

---

With descriptive filenames,

```text
FMR_1_R_0.1_E_linear_EP_0.01.csv

FMR_3_R_10_E_full_EP_0.10.csv

FMR_5_R_100_E_circular_EP_0.01.csv
```

the complete configuration is immediately visible.

No need to open the file.

The filename itself documents the experiment.

---

# Research Workflow

The workflow becomes:

```text
Choose Hyperparameters

↓

Feature Map Repetitions

↓

Regularization Parameter (C)

↓

Entanglement Type

↓

Epsilon (ε)

↓

Construct Descriptive Filename

↓

Save Results Automatically

↓

Repeat Hundreds of Times
```

Every experiment produces a uniquely named file.

---

# Why Researchers Do This

Suppose one experiment achieves the lowest prediction error.

Several weeks later,

you want to reproduce those results.

If the filename is

```text
result.csv
```

you have no idea which hyperparameters were used.

However, if the filename is

```text
FMR_3_R_10_E_full_EP_0.01.csv
```

you immediately know:

```text
Feature Map Repetitions = 3

↓

Regularization Parameter = 10

↓

Entanglement = Full

↓

Epsilon = 0.01
```

Reproducing the experiment becomes straightforward.

---

# Good Computational Research Practice

This cell demonstrates an important principle of scientific computing:

> **Metadata should travel with the results.**

Here,

the metadata (hyperparameters) are embedded directly into the filename.

Even if someone copies the CSV file to another computer,

the filename still describes how the experiment was generated.

This greatly improves:

- Reproducibility
- Traceability
- Collaboration
- Long-term project organization

---

# Example

Suppose three experiments are performed.

Instead of

```text
result1.csv

result2.csv

result3.csv
```

the notebook could produce

```text
FMR_1_R_0.1_E_linear_EP_0.01.csv

FMR_3_R_10_E_full_EP_0.10.csv

FMR_5_R_100_E_circular_EP_0.01.csv
```

Months later,

the researcher can immediately identify the complete hyperparameter configuration without opening the files.

---

# Research Insight

Notice that this cell does not change:

- the quantum feature map,
- the fidelity quantum kernel,
- the QSVR model,
- or the machine learning algorithm.

Its sole purpose is **experiment organization**.

As research projects grow larger,

organization becomes increasingly important.

Many experienced researchers spend as much time designing reproducible workflows as they do developing new algorithms.

Automatic filename generation is one small but valuable example of that philosophy.

---

# Overall Workflow

```text
Choose Hyperparameters

↓

Feature Map Repetitions

↓

Regularization Parameter (C)

↓

Entanglement Type

↓

Epsilon (ε)

↓

Construct Descriptive Filename

↓

Save Results Automatically

↓

Repeat for Every Experiment
```

---

# Key Takeaways

- This cell is **not part of the QSVR algorithm**; it is part of the experiment management workflow.
- Its purpose is to generate descriptive filenames that uniquely identify each experiment.
- The filename embeds important hyperparameters such as:
  - **Feature Map Repetitions (FMR)**
  - **Regularization Parameter (R or C)**
  - **Entanglement Topology (E)**
  - **Epsilon (EP or ε)**
- Including **epsilon** is the major difference compared with the QSVC notebook because ε is a regression-specific hyperparameter.
- Descriptive filenames improve traceability, reproducibility, and organization by allowing researchers to identify an experiment without opening the corresponding file.
- Embedding metadata into filenames is a common and highly recommended practice in computational research, especially when running large batches of experiments.



# The Heart of the QSVR Code

This section is the **core of the entire notebook**.

Everything before this point has been preparing the data and defining helper functions.

Now, the actual Quantum Machine Learning experiment begins.

This section contains:

- Cross Validation
- Hyperparameter Search
- Quantum Model Construction
- Model Training
- Prediction
- Result Collection
- Saving Results

This is where the paper spends most of its computational time.

---

# What Is the Code Trying to Answer?

The central research question is:

> **Which Quantum Support Vector Regression (QSVR) configuration predicts the stacking fault energy (SFE) of Mg alloys most accurately?**

Instead of building a single model, the notebook systematically evaluates many different QSVR models.

For every experiment, it varies several hyperparameters:

- Different **Regularization Parameters (C)**
- Different **Feature Map Repetitions (reps)**
- Different **Epsilon (ε) values**
- Different **Entanglement Topologies**

Each combination produces a completely new quantum regression model.

The notebook then compares their predictive performance to determine which configuration performs best.

---

# Why Test So Many Models?

There is no theoretical guarantee that one particular set of hyperparameters will perform best.

For example:

- A larger **C** may fit the training data more closely but risk overfitting.
- More **feature map repetitions** increase the expressive power of the quantum circuit but also increase circuit complexity.
- Different **entanglement patterns** create different quantum feature spaces.
- Different **epsilon (ε)** values change how tolerant the regression model is to small prediction errors.

Since the best combination is unknown beforehand, the notebook performs a **hyperparameter search**, testing every possible combination.

---

# The Big Picture

The overall workflow of the experiment is:

```text
Entire Dataset
      │
      ▼
Cross Validation
      │
      ▼
Training / Testing Split
      │
      ▼
Scale Features
      │
      ▼
Try Every C
      │
      ▼
Try Every Feature Map Repetition
      │
      ▼
Try Every ε (Epsilon)
      │
      ▼
Try Every Entanglement Topology
      │
      ▼
Build Quantum Kernel
      │
      ▼
Create QSVR
      │
      ▼
Train Model
      │
      ▼
Predict Stacking Fault Energy
      │
      ▼
Evaluate Performance
      │
      ▼
Store Results
      │
      ▼
Save CSV
      │
      ▼
Repeat for Next Split
```

Each pass through this workflow represents one complete QSVR experiment.

---

# What Happens Internally?

For every train-test split, the notebook performs the following steps:

1. Prepare the training and testing datasets.
2. Scale the numerical features.
3. Build a new **ZZFeatureMap** using the selected hyperparameters.
4. Construct a **FidelityQuantumKernel**.
5. Create a **QSVR** model.
6. Train the model using the training alloys.
7. Predict the stacking fault energies of both the training and unseen test alloys.
8. Store all predictions and evaluation metrics.
9. Save the updated results to a CSV file.

This entire process is repeated for every hyperparameter combination.

---

# Difference Compared with QSVC

This workflow is **almost identical** to the QSVC notebook.

The main difference is the addition of **one extra hyperparameter loop**.

QSVC searches over:

```text
C

↓

Feature Map Repetitions

↓

Entanglement
```

QSVR searches over:

```text
C

↓

Feature Map Repetitions

↓

Epsilon (ε)

↓

Entanglement
```

The epsilon parameter is unique to **Support Vector Regression** because regression predicts continuous values rather than discrete classes.

---

# Why Is This Computationally Expensive?

Each hyperparameter combination requires:

- constructing a new quantum feature map,
- computing a new quantum kernel,
- training a new QSVR model,
- making predictions,
- and recording the results.

Since this is repeated across all cross-validation splits and all hyperparameter combinations, the total number of QSVR models trained can easily reach **tens of thousands**.

This exhaustive search helps ensure that the reported performance is not due to a fortunate choice of parameters but reflects a systematic evaluation of the model.

---

# Research Insight

This section represents the **experimental engine** of the paper.

Everything defined earlier—dataset preparation, scaling, feature maps, kernels, and training functions—is brought together here into one automated workflow.

Rather than relying on intuition to choose hyperparameters, the authors allow the computer to systematically evaluate every combination.

This approach improves the reliability, reproducibility, and scientific rigor of the study.

---

# Key Takeaways

- This is the **core computational section** of the notebook.
- It performs the complete QSVR experiment from data preparation to result storage.
- The notebook systematically searches over:
  - **Regularization Parameter (C)**
  - **Feature Map Repetitions (reps)**
  - **Epsilon (ε)**
  - **Entanglement Topology**
- Every hyperparameter combination results in a completely new QSVR model.
- The entire workflow is repeated for every cross-validation split, ensuring a rigorous evaluation.
- Compared with QSVC, the only structural difference is the additional **epsilon (ε) loop**, which is specific to Support Vector Regression.



# The First Loop – Cross Validation

The first loop begins the actual machine learning experiment.

```python
for train_indices, test_indices in rkf.split(X):
```

Here, `rkf` is the **RepeatedKFold** object created earlier.

It does not contain the training data itself.

Instead, it acts as a **cross-validation planner**, deciding which samples will be used for training and which will be reserved for testing during each iteration.

Every iteration of this loop produces two arrays:

- **Training indices**
- **Testing indices**

These indices tell Python exactly which rows of the dataset belong to the training set and which belong to the test set.

---

# Example

Suppose the dataset contains 21 alloys.

One iteration of Leave-One-Out Cross Validation (LOOCV) might produce:

```text
Training Indices

[0,1,2,3,4,5,6,7,8,9,
10,11,12,13,14,15,16,17,18,20]
```

```text
Testing Indices

[19]
```

This means:

- Alloy **19** is held out as the unseen test sample.
- The remaining **20 alloys** are used for training.

On the next iteration, a different alloy becomes the test sample.

Eventually, every alloy is tested exactly once in one complete cross-validation cycle.

---

# Preparing the Dataset

Inside the loop, the notebook calls

```python
X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(
    X, y, train_indices, test_indices
)
```

This function performs all of the preprocessing required before training the quantum model.

Internally, it carries out several tasks:

1. Splits the dataset into training and testing subsets.
2. Separates the element names from the numerical features.
3. Removes the element names from the feature matrix.
4. Scales the numerical features using **MinMaxScaler**.
5. Returns all processed data needed for the experiment.

Rather than repeating these steps inside the main loop, the notebook places them in a separate function, making the code cleaner and easier to maintain.

---

# What Does the Function Return?

The function returns six objects.

## 1. Training Features

```text
X_train
```

These are the numerical input features used to train the QSVR model.

For this project, each training sample contains:

- Electronegativity
- Bulk Modulus
- Atomic Volume

After preprocessing, the values have already been scaled.

---

## 2. Training Labels

```text
y_train
```

These are the **true stacking fault energy (SFE)** values corresponding to each training alloy.

Unlike QSVC, these remain **continuous numerical values** because QSVR is a regression algorithm.

Example:

```text
34.8

18.2

42.1

25.7
```

These are the target values the regression model attempts to learn.

---

## 3. Testing Features

```text
X_test
```

These contain the numerical features of the alloy reserved for testing.

The model never sees this sample during training.

It is used only to evaluate how well the trained model generalizes to unseen data.

---

## 4. Testing Labels

```text
y_test
```

These are the true SFE values for the testing alloy.

After prediction, the notebook compares the predicted value against these true values to evaluate the model's performance.

---

## 5. Training Elements

```text
element_train
```

This contains the names of the alloys in the training set.

For example:

```text
Mg-Al

Mg-Zn

Mg-Y
```

These names are removed from the numerical calculations but retained for recording and interpreting the experimental results.

---

## 6. Testing Elements

```text
element_test
```

This stores the name of the alloy currently being tested.

For example:

```text
Mg-Zn
```

Later, this information is written to the results table so researchers know exactly which alloy was predicted in each experiment.

---

# Why Use a Separate Function?

The preprocessing steps—splitting, removing element names, and scaling—are required every time a new train-test split is created.

Encapsulating these operations in a dedicated function provides several advantages:

- The main training loop remains concise and easy to read.
- Code duplication is avoided.
- Any changes to preprocessing need to be made in only one place.
- The workflow becomes easier to debug and maintain.

This is an example of good software engineering practice in scientific computing.

---

# Workflow So Far

```text
Dataset
      │
      ▼
RepeatedKFold
      │
      ▼
Generate Train/Test Indices
      │
      ▼
prepare_dataset_k_fold()
      │
      ├── Split Dataset
      ├── Remove Element Names
      ├── Scale Features
      └── Return Processed Data
      │
      ▼
Ready for QSVR Training
```

---

# Research Insight

This loop is responsible for ensuring a **fair evaluation** of the regression model.

Every alloy in the dataset is treated as an unseen sample at some point during cross-validation.

By repeating this process across all samples (and multiple repetitions if specified), the notebook obtains a more reliable estimate of the model's predictive performance than a single train-test split could provide.

---

# Key Takeaways

- The first loop begins the cross-validation process.
- `RepeatedKFold` generates the training and testing indices for each iteration.
- `prepare_dataset_k_fold()` prepares the data by:
  - Splitting the dataset,
  - Removing element names from the feature matrix,
  - Scaling the numerical features,
  - Returning processed training and testing data.
- The function returns:
  - **Training Features (`X_train`)**
  - **Training Targets (`y_train`)**
  - **Testing Features (`X_test`)**
  - **Testing Targets (`y_test`)**
  - **Training Element Names (`element_train`)**
  - **Testing Element Names (`element_test`)**
- This preprocessing ensures that each QSVR model is trained and evaluated consistently across all cross-validation folds.


# The Second Loop – Loop Over the Regularization Parameter (C)

The second loop performs a **hyperparameter search** over the **regularization parameter (C)**.

```python
for C_value in REGU_PARA_LIST:
```

Earlier in the notebook, we defined:

```python
REGU_PARA_LIST = [0.1, 1, 10, 100]
```

Instead of assuming one value of **C** is optimal, the notebook systematically tests every value in this list.

Therefore, this loop performs **four separate experiments** for every train-test split.

```text
C = 0.1

      ↓

C = 1

      ↓

C = 10

      ↓

C = 100
```

Each value results in a **new QSVR model** being constructed and evaluated.

---

# What Is the Regularization Parameter (C)?

The parameter **C** originates from the classical **Support Vector Machine (SVM)** framework and is also used in **Support Vector Regression (SVR)**.

It controls the balance between:

- fitting the training data closely, and
- keeping the regression model simple enough to generalize well to unseen data.

In other words, **C determines how strongly the model is penalized for prediction errors.**

---

# Small C

A **small value of C** places greater emphasis on regularization.

For example,

```text
C = 0.1
```

The model becomes more tolerant of prediction errors.

Instead of trying to fit every training sample as closely as possible, it prefers a smoother and simpler regression function.

Advantages:

- Lower risk of overfitting
- Better generalization on unseen data

Disadvantage:

- May underfit if the model becomes too simple.

---

# Large C

A **large value of C** reduces the strength of regularization.

For example,

```text
C = 100
```

Now the regression model attempts to fit the training data much more closely.

Advantages:

- Lower training error
- Better fit to complex relationships

Disadvantages:

- Higher risk of overfitting
- May perform worse on new, unseen alloys.

---

# Why Doesn't the Paper Choose One Value?

Before running the experiments, the authors do not know which value of **C** will provide the best predictive performance.

Rather than relying on intuition, they let the computer test each possibility.

This process is called a **hyperparameter search**.

The notebook evaluates:

```text
C = 0.1

↓

C = 1

↓

C = 10

↓

C = 100
```

Later, the results from all four settings can be compared using performance metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The value that gives the best overall performance can then be selected.

---

# Relationship Between C and Model Complexity

A useful way to think about **C** is:

```text
Small C
      │
      ▼
More Regularization
      │
      ▼
Simpler Model
      │
      ▼
Lower Risk of Overfitting
```

```text
Large C
      │
      ▼
Less Regularization
      │
      ▼
More Complex Model
      │
      ▼
Higher Risk of Overfitting
```

The goal is to find a balance where the model captures the underlying relationship without simply memorizing the training data.

---

# What Happens Inside the Loop?

For every train-test split generated by cross-validation, the notebook repeats the entire experiment four times.

For example:

```text
Training/Test Split

↓

C = 0.1

↓

Build QSVR

↓

Train

↓

Predict

↓

Store Results

↓

C = 1

↓

Build New QSVR

↓

Train Again

↓

Predict Again

↓

Store Results

↓

Repeat for C = 10

↓

Repeat for C = 100
```

Notice that each value of **C** creates a **completely new regression model**.

The notebook does **not** modify the existing model.

Instead, it builds, trains, and evaluates a fresh QSVR model for every hyperparameter value.

---

# Research Insight

Testing multiple values of **C** is a standard practice in Support Vector Machines and Support Vector Regression.

The optimal amount of regularization depends on the dataset.

A value that performs well on one materials dataset may perform poorly on another.

By evaluating multiple values systematically, the authors reduce the likelihood that their conclusions depend on an arbitrary hyperparameter choice.

---

# Key Takeaways

- The second loop performs a **hyperparameter search** over the regularization parameter **C**.
- `REGU_PARA_LIST = [0.1, 1, 10, 100]`, so the notebook evaluates **four different QSVR models** for every train-test split.
- **Small C** increases regularization, producing a simpler model that is more tolerant of prediction errors.
- **Large C** decreases regularization, allowing the model to fit the training data more closely but increasing the risk of overfitting.
- Each value of **C** results in a completely new QSVR model being constructed, trained, evaluated, and recorded.
- This systematic search helps identify the regularization strength that provides the best predictive performance for stacking fault energy regression.



# The Third Loop – Loop Over Feature Map Repetitions

The third loop performs a **hyperparameter search** over the number of **feature map repetitions**.

```python
for feature_map_reps in FEATURE_MAP_REPS_LIST:
```

Earlier in the notebook, we defined:

```python
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
```

Rather than assuming one quantum circuit depth is optimal, the notebook systematically evaluates **five different feature map depths**.

This means that for **every value of C**, the experiment is repeated five times using different numbers of feature map repetitions.

```text
reps = 1

      ↓

reps = 2

      ↓

reps = 3

      ↓

reps = 4

      ↓

reps = 5
```

Each value produces a **new quantum feature map**, which in turn creates a **new quantum kernel** and ultimately a **new QSVR model**.

---

# What Does "reps" Mean?

The parameter **reps** specifies how many times the feature map circuit is repeated.

Recall that a **feature map** is responsible for encoding classical numerical data into a quantum state.

```text
Classical Features

↓

Feature Map

↓

Quantum State

↓

Quantum Kernel

↓

QSVR
```

Increasing the number of repetitions means applying the same encoding circuit multiple times.

---

# Why Repeat the Feature Map?

A single application of the feature map creates one layer of quantum operations.

```text
Input Data

↓

Feature Map

↓

Quantum State
```

If the feature map is repeated,

```text
Input Data

↓

Feature Map

↓

Feature Map

↓

Feature Map

↓

Quantum State
```

the resulting quantum circuit becomes deeper and can represent more complex transformations of the input features.

---

# Small Number of Repetitions

For example,

```text
reps = 1
```

The quantum circuit is relatively shallow.

Advantages:

- Faster to simulate
- Less computational cost
- Lower chance of introducing unnecessary complexity

However,

- the circuit may not capture complicated relationships between features.

---

# Large Number of Repetitions

For example,

```text
reps = 5
```

The same encoding circuit is repeated five times.

Advantages:

- Produces a richer quantum feature space.
- Can capture more complex relationships between material descriptors.

Disadvantages:

- Larger quantum circuits.
- Higher computational cost.
- May increase the risk of overfitting, especially for very small datasets.

---

# Why Doesn't the Paper Use One Value?

The authors do not know beforehand how much circuit depth is appropriate for this materials dataset.

A shallow circuit may be too simple.

A deep circuit may become unnecessarily complex.

Therefore, instead of making assumptions, the notebook evaluates several possibilities:

```text
reps = 1

↓

reps = 2

↓

reps = 3

↓

reps = 4

↓

reps = 5
```

The best-performing value can later be identified by comparing the regression metrics across all experiments.

---

# Effect on the Quantum Circuit

Suppose the original feature map is represented as one block.

With different values of `reps`, the circuit becomes:

```text
reps = 1

[ Feature Map ]
```

```text
reps = 2

[ Feature Map ]

↓

[ Feature Map ]
```

```text
reps = 5

[ Feature Map ]

↓

[ Feature Map ]

↓

[ Feature Map ]

↓

[ Feature Map ]

↓

[ Feature Map ]
```

Each additional repetition increases the circuit depth and changes how the classical features are embedded into the quantum state.

---

# What Happens Inside the Loop?

For every value of **C**, the notebook now repeats the experiment five times.

For example:

```text
Choose C

↓

reps = 1

↓

Build ZZFeatureMap

↓

Build Quantum Kernel

↓

Build QSVR

↓

Train

↓

Predict

↓

Store Results

↓

reps = 2

↓

Build New ZZFeatureMap

↓

Build New Quantum Kernel

↓

Build New QSVR

↓

Repeat...
```

Each value of `reps` creates a completely **new quantum feature map**.

Consequently, the quantum kernel changes, and a completely new QSVR model is trained.

---

# Total Number of Experiments So Far

At this point in the nested loops, we have:

- **4 values of C**
- **5 values of feature map repetitions**

Therefore,

```text
4 × 5 = 20
```

different QSVR configurations are evaluated for **every train-test split**, before considering the remaining hyperparameters such as **epsilon** and **entanglement**.

---

# Research Insight

One of the unique aspects of Quantum Machine Learning is that the **feature map itself is a tunable hyperparameter**.

In classical machine learning, feature engineering often happens before model training.

In quantum machine learning, however, the encoding circuit directly determines the quantum feature space.

Changing `reps` changes how the data is represented in that space, which can significantly affect the performance of the quantum kernel and, ultimately, the regression model.

This is why the notebook systematically evaluates multiple circuit depths instead of relying on a single design.

---

# Key Takeaways

- The third loop performs a **hyperparameter search** over the number of **feature map repetitions (`reps`)**.
- `FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]`, so **five different circuit depths** are tested.
- Increasing `reps` makes the quantum circuit deeper and produces a richer quantum feature space.
- Deeper circuits may capture more complex relationships between material features but also increase computational cost and the potential for overfitting.
- Every value of `reps` creates a **new ZZFeatureMap**, which leads to a **new Fidelity Quantum Kernel** and a **new QSVR model**.
- Combined with the previous loop over **C**, the notebook now evaluates **20 different QSVR configurations** for every train-test split.



# The Fourth Loop – Loop Over Epsilon (ε)

The fourth loop performs a **hyperparameter search** over the **epsilon (ε)** parameter of the Quantum Support Vector Regressor (QSVR).

```python
for epsilon_value in EPSILON_LIST:
```

Earlier in the notebook, we defined:

```python
EPSILON_LIST = [0.01, 0.001]
```

Unlike QSVC, **QSVR introduces an additional hyperparameter called epsilon (ε)**.

Instead of assuming one value is best, the notebook evaluates **two different epsilon values**.

```text
ε = 0.01

      ↓

ε = 0.001
```

For every combination of **C** and **feature map repetitions**, the notebook builds and trains **two separate QSVR models**, one for each epsilon value.

---

# What Is Epsilon (ε)?

Epsilon is one of the defining ideas behind **Support Vector Regression (SVR)**.

Unlike classification, where predictions are simply **correct** or **incorrect**, regression predicts **continuous numerical values**.

For example,

```text
Actual SFE = 18.42

Predicted SFE = 18.40
```

The prediction is not exactly correct, but it is extremely close.

Should the model be penalized for such a tiny error?

SVR answers this question using **epsilon (ε)**.

---

# The ε-Insensitive Tube

Rather than demanding perfect predictions, SVR creates a small region around every target value called the **epsilon-insensitive tube**.

```text
                 ε Tube

           -------------------
           |                 |
           |   Actual Value  |
           |        ●         |
           |                 |
           -------------------
```

If the prediction falls **inside** this tube,

```text
Prediction

↓

Inside ε Tube

↓

No Penalty
```

the model considers the prediction "good enough."

Only predictions **outside** the tube contribute to the training loss.

---

# Example

Suppose:

```text
Actual SFE = 20.00
```

If

```text
ε = 0.01
```

then the acceptable region becomes

```text
19.99  ←──────── 20.00 ───────→ 20.01
```

A prediction of

```text
20.005
```

lies inside the tube.

```text
Prediction Error = 0.005

↓

Error < ε

↓

No Penalty
```

However,

```text
Prediction = 20.03
```

produces

```text
Error = 0.03

↓

Error > ε

↓

Penalty Applied
```

---

# Large Epsilon

A **larger epsilon** creates a wider tube.

For example,

```text
ε = 0.01
```

```text
Wide Tube

↓

More Errors Ignored

↓

Simpler Regression Function
```

Advantages:

- More tolerant of small prediction errors.
- Often produces smoother models.
- Less sensitive to noise in the data.

Disadvantage:

- May ignore meaningful differences if the tube becomes too wide.

---

# Small Epsilon

A **smaller epsilon** creates a much narrower tube.

For example,

```text
ε = 0.001
```

```text
Narrow Tube

↓

Very Few Errors Ignored

↓

Model Must Fit Data More Closely
```

Advantages:

- Higher precision.
- Better ability to capture subtle trends.

Disadvantages:

- More sensitive to noise.
- Increased risk of overfitting.

---

# Why Test Two Values?

The authors do not know beforehand which epsilon value will produce the best regression model for predicting stacking fault energy.

Therefore, instead of selecting one value arbitrarily, they evaluate both:

```text
ε = 0.01

↓

Train QSVR

↓

Evaluate

↓

ε = 0.001

↓

Train New QSVR

↓

Evaluate
```

The results can later be compared using regression metrics such as:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

The epsilon value that provides the best predictive performance can then be selected.

---

# What Happens Inside the Loop?

At this stage of the notebook, the loops have already selected:

- one value of **C**
- one value of **feature map repetitions**

Now the notebook repeats the experiment twice.

```text
Choose C

↓

Choose Feature Map Repetitions

↓

ε = 0.01

↓

Build QSVR

↓

Train

↓

Predict

↓

Store Results

↓

ε = 0.001

↓

Build New QSVR

↓

Train Again

↓

Predict Again

↓

Store Results
```

Notice that changing **epsilon** creates a **new regression model** because the optimization problem solved by SVR depends directly on the chosen ε value.

---

# Total Number of Experiments So Far

Up to this point, the notebook has explored:

- **4 values of C**
- **5 values of feature map repetitions**
- **2 values of epsilon**

Therefore,

```text
4 × 5 × 2 = 40
```

different QSVR configurations are evaluated for **every train-test split**, before considering the final hyperparameter: **entanglement**.

---

# Research Insight

The epsilon parameter exists **only in Support Vector Regression**.

It has no equivalent in QSVC because classification predicts discrete class labels rather than continuous numerical values.

In regression, however, not every tiny prediction error is meaningful.

The ε-insensitive loss function allows the model to ignore very small deviations while focusing on larger prediction errors that matter more.

Choosing an appropriate epsilon is therefore an important part of building an accurate and robust regression model.

---

# Key Takeaways

- The fourth loop performs a **hyperparameter search** over the **epsilon (ε)** parameter of QSVR.
- `EPSILON_LIST = [0.01, 0.001]`, so **two different epsilon values** are evaluated.
- Epsilon defines the **ε-insensitive tube**, where prediction errors smaller than ε are ignored during training.
- **Larger ε** creates a wider tolerance region, making the model more forgiving of small errors.
- **Smaller ε** creates a narrower tolerance region, forcing the model to fit the data more precisely but increasing sensitivity to noise.
- Combined with the previous loops over **C** and **feature map repetitions**, the notebook now evaluates **40 different QSVR configurations** for every train-test split.



# The Fifth Loop – Loop Over Entanglement

The fifth loop performs a **hyperparameter search** over the **entanglement topology** used in the quantum feature map.

```python
for entanglement in ENTANGLEMENT_LIST:
```

Earlier in the notebook, we defined:

```python
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']
```

Instead of assuming one entanglement pattern is best, the notebook systematically evaluates **three different quantum circuit topologies**.

```text
Linear

      ↓

Full

      ↓

Circular
```

For every combination of **C**, **feature map repetitions**, and **epsilon**, the notebook constructs and trains **three separate QSVR models**, each using a different entanglement strategy.

---

# What Is Entanglement?

The **ZZFeatureMap** does more than encode each feature independently.

It also creates **interactions between qubits**.

These interactions are called **entanglement**.

Instead of each qubit acting independently,

```text
Q0    Q1    Q2
```

the qubits can exchange information during the feature encoding process.

The way these connections are arranged is determined by the **entanglement topology**.

Changing this topology changes the quantum feature space, which may influence how well the quantum kernel distinguishes different alloys.

---

# Linear Entanglement

```text
Q0 ─── Q1 ─── Q2
```

Each qubit communicates only with its immediate neighbour.

Advantages:

- Simplest topology.
- Lowest computational cost.
- Shallower quantum circuits.

Disadvantages:

- May not capture complex relationships between distant features.

---

# Full Entanglement

```text
      Q0
     /  \
    /    \
   Q1────Q2
```

Every qubit interacts with every other qubit.

Advantages:

- Richest quantum feature space.
- Can represent more complex feature interactions.

Disadvantages:

- Deepest circuits.
- Highest computational cost.
- More difficult to execute on real quantum hardware.

---

# Circular Entanglement

```text
Q0 ─── Q1
│       │
└── Q2 ─┘
```

The qubits form a closed ring.

Advantages:

- More expressive than linear entanglement.
- Requires fewer connections than full entanglement.
- Provides a balance between circuit complexity and expressive power.

---

# Why Test Different Topologies?

The authors do not know beforehand which interaction pattern is most suitable for predicting stacking fault energy.

Different datasets benefit from different quantum feature spaces.

Instead of choosing one topology arbitrarily, the notebook evaluates all three.

```text
Linear

↓

Train QSVR

↓

Evaluate

↓

Full

↓

Train New QSVR

↓

Evaluate

↓

Circular

↓

Train New QSVR

↓

Evaluate
```

The best-performing topology can later be identified by comparing the regression metrics across all experiments.

---

# What Happens Inside the Loop?

By the time execution reaches this loop, the notebook has already selected:

- one value of **C**
- one value of **feature map repetitions**
- one value of **epsilon**

Now it repeats the experiment for each entanglement strategy.

```text
Choose C

↓

Choose Feature Map Repetitions

↓

Choose Epsilon

↓

Linear

↓

Build ZZFeatureMap

↓

Build Quantum Kernel

↓

Build QSVR

↓

Train

↓

Predict

↓

Store Results

↓

Full

↓

Repeat

↓

Circular

↓

Repeat
```

Every entanglement topology creates a **new ZZFeatureMap**.

Since the feature map changes, the **Fidelity Quantum Kernel** also changes, requiring a completely new QSVR model to be trained.

---

# Total Number of Experiments

The notebook now evaluates:

- **4 values of C**
- **5 values of feature map repetitions**
- **2 values of epsilon**
- **3 entanglement topologies**

Therefore,

```text
4 × 5 × 2 × 3 = 120
```

different QSVR configurations are evaluated for **every train-test split**.

The dataset contains **21 alloys**, and Leave-One-Out Cross Validation tests each alloy once.

```text
21 × 120 = 2,520
```

QSVR models are trained during a single repetition of cross-validation.

Finally,

```text
N_REPEATS = 10
```

means the entire Leave-One-Out procedure is repeated ten times.

```text
2,520 × 10 = 25,200
```

**QSVR models** are trained throughout the complete experiment.

---

# Monitoring the Experiment

```python
print(f'C:{C_value} feature_map_reps:{feature_map_reps} entanglement:{entanglement}')
```

This statement simply prints the current hyperparameter combination being evaluated.

For example,

```text
C:10 feature_map_reps:3 entanglement:full
```

This makes it easy to monitor long-running experiments and identify which configuration is currently being processed.

---

# Building the QSVR Model

```python
qsvr = reconfig_quantum_kernel_qsvr(
    feature_dimension=NUM_FEATURES,
    C=C_value,
    reps=feature_map_reps,
    epsilon=epsilon_value,
    entangle=entanglement
)
```

We examined this function earlier.

Internally, it performs:

```text
ZZFeatureMap

↓

Fidelity Quantum Kernel

↓

QSVR
```

Each iteration constructs a completely **new quantum regression model** with the current set of hyperparameters.

---

# Training and Prediction

```python
predict_train, predict_test = train_qsvr(
    qsvr,
    X_train,
    y_train,
    X_test
)
```

This function trains the newly created QSVR model using the training data.

Once training is complete, the model predicts:

- the training samples (to assess how well the model learned),
- the unseen test sample (to evaluate generalization).

---

# Converting Predictions Back to Their Original Scale

Earlier in the notebook, the target values were scaled.

```python
all_preds = y_scaler.inverse_transform(all_preds.reshape(-1, 1))
all_targets = y_scaler.inverse_transform(all_targets.reshape(-1, 1))
```

These lines convert both the predictions and the true target values back to their **original stacking fault energy units**.

```text
Scaled Predictions

↓

Inverse Scaling

↓

Predicted SFE (mJ/m²)
```

This makes the saved results directly interpretable.

---

# Saving the Results

Each experiment is summarized in a dictionary.

```python
new_row = {
    'C': C_value,
    'reps': feature_map_reps,
    'epsilon': epsilon_value,
    'entanglement': entanglement,
    ...
}
```

This dictionary records:

- Regularization parameter (C)
- Feature map repetitions
- Epsilon
- Entanglement topology
- Test alloy
- Predicted stacking fault energy
- Actual stacking fault energy
- Training predictions
- Training targets
- Performance metrics

Every experiment contributes one new row to the results table.

---

# Updating the DataFrame

```python
df.loc[len(df)] = new_row
```

This appends the current experiment to the end of the DataFrame.

As the notebook runs,

```text
Experiment 1

↓

Experiment 2

↓

Experiment 3

↓

...

↓

25,200 Experiments

↓

Complete Results Table
```

---

# Saving Progress Continuously

```python
df.to_csv(file_name, index=False)
```

After every experiment, the updated DataFrame is written to a CSV file.

This is an excellent research practice because:

- progress is not lost if the program stops unexpectedly,
- partial results remain available,
- long-running experiments can be resumed or inspected at any time.

---

# Additional Metadata

```python
df.at[0, "info"]
```

This line stores additional descriptive information in the DataFrame.

Rather than representing a prediction, it is typically used to record metadata such as:

- experiment settings,
- dataset information,
- notes about the current run.

This makes the output files more self-documenting and easier to interpret later.

---

# Research Insight

This loop completes the **full hyperparameter search**.

By combining:

- regularization (**C**),
- feature map depth (**reps**),
- regression tolerance (**ε**),
- and quantum circuit topology (**entanglement**),

the notebook performs an exhaustive exploration of the quantum model design space.

Although computationally expensive, this systematic approach ensures that the reported results are based on a comprehensive evaluation rather than a single arbitrary choice of hyperparameters.

---

# Key Takeaways

- The fifth loop performs a **hyperparameter search** over the **entanglement topology** of the quantum feature map.
- Three topologies are evaluated: **linear**, **full**, and **circular**.
- Every entanglement pattern produces a new **ZZFeatureMap**, **Fidelity Quantum Kernel**, and **QSVR** model.
- Combined with the previous loops, the notebook evaluates **120 unique QSVR configurations** for every Leave-One-Out split.
- With **21 test splits** and **10 repetitions**, the complete experiment trains **25,200 QSVR models**.
- Each experiment's predictions, hyperparameters, and performance metrics are recorded in a DataFrame and continuously saved to a CSV file, ensuring reproducibility and safeguarding progress during long-running computations.



# Entire Pipeline of the Experiment

This flowchart summarizes the complete workflow followed by the **Quantum Support Vector Regressor (QSVR)** experiment, from loading the dataset to saving the final results.

```text
Dataset
   │
   ▼
Leave-One-Out Cross Validation (LOOCV)
   │
   ▼
Split Training and Testing Sets
   │
   ▼
Loop over Regularization Parameter (C)
   │
   ▼
Loop over Feature Map Repetitions (Reps)
   │
   ▼
Loop over Epsilon (ε)
   │
   ▼
Loop over Entanglement Strategy
   │
   ▼
Build Quantum Kernel
   │
   ▼
Create QSVR Model
   │
   ▼
Train Model
   │
   ▼
Predict Continuous Stacking Fault Energy (SFE)
   │
   ▼
Inverse Scaling
   │
   ▼
Store Results
   │
   ▼
Save Results to CSV
   │
   ▼
Repeat for the Next Hyperparameter Combination and Cross-Validation Split
```

---

## Pipeline Explanation

The experiment follows a systematic workflow designed to evaluate many different QSVR configurations.

1. **Load the Dataset**
   - Read the alloy dataset containing the input features and target stacking fault energies.

2. **Leave-One-Out Cross Validation (LOOCV)**
   - Since the dataset contains only 21 alloys, one alloy is held out for testing while the remaining 20 are used for training.
   - This process is repeated until every alloy has served as the test sample.

3. **Split the Dataset**
   - Separate the current fold into training and testing sets.
   - Remove non-numerical columns (such as alloy names) and scale the numerical features.

4. **Hyperparameter Search**
   - Iterate through every combination of:
     - Regularization parameter (**C**)
     - Feature map repetitions (**Reps**)
     - Epsilon (**ε**)
     - Entanglement topology

5. **Build the Quantum Kernel**
   - Construct a `ZZFeatureMap`.
   - Use it to create a `FidelityQuantumKernel`.

6. **Create the QSVR Model**
   - Initialize a new Quantum Support Vector Regressor using the selected hyperparameters and quantum kernel.

7. **Train the Model**
   - Fit the QSVR using the training data.

8. **Predict Continuous Stacking Fault Energy**
   - Generate predictions for both the training and unseen testing samples.
   - Unlike QSVC, these predictions are continuous numerical values rather than class labels.

9. **Inverse Scaling**
   - Convert the predicted values back to their original physical scale for interpretation.

10. **Store Results**
    - Record the hyperparameters, predictions, actual values, and performance metrics in a DataFrame.

11. **Save Results**
    - Continuously update the experiment CSV file so that progress is preserved even if execution is interrupted.

12. **Repeat**
    - Continue until every combination of hyperparameters has been evaluated for every cross-validation split.

---

# Key Takeaway

This pipeline combines:

- Leave-One-Out Cross Validation (LOOCV)
- Exhaustive Hyperparameter Search
- Quantum Kernel Construction
- QSVR Training
- Continuous Property Prediction
- Automatic Result Logging

to systematically determine which quantum regression model best predicts the **Stacking Fault Energy (SFE)** of magnesium alloys.



# Final Conclusions

The two quantum machine learning approaches studied in this work solve different types of prediction problems, even though they share most of their underlying workflow.

---

## QSVC vs QSVR

### Quantum Support Vector Classifier (QSVC)

QSVC answers the question:

> **"Is this alloy's Stacking Fault Energy (SFE) above or below a predefined threshold?"**

This is a **binary classification** problem.

Instead of predicting the exact SFE value, the model assigns each alloy to one of two classes (e.g., low-SFE or high-SFE) based on a chosen threshold.

---

### Quantum Support Vector Regressor (QSVR)

QSVR answers the question:

> **"What is this alloy's actual Stacking Fault Energy (SFE)?"**

This is a **regression** problem.

Rather than assigning a class label, the model predicts the continuous numerical value of the stacking fault energy.

---

# Similarities Between QSVC and QSVR

QSVC and QSVR are remarkably similar, with approximately **90–95% of the overall workflow being identical**.

Both approaches use:

- The same dataset and input features
- The same data preprocessing and feature scaling
- Leave-One-Out Cross Validation (LOOCV)
- The same `ZZFeatureMap` for encoding classical data into quantum states
- The same `FidelityQuantumKernel`
- Hyperparameter searches over:
  - Regularization parameter (**C**)
  - Feature map repetitions (**Reps**)
  - Entanglement topology
- Automated experiment management
- Continuous result logging and CSV generation

As a result, much of the code is shared between the two implementations.

---

# Key Differences

Although the overall workflow is very similar, the learning objective differs.

## QSVC

QSVC transforms the continuous stacking fault energy values into **binary classes** using a predefined threshold.

The model is then trained as a **classifier** to determine whether an alloy belongs to the low-SFE or high-SFE category.

---

## QSVR

QSVR retains the original continuous stacking fault energy values.

Instead of classifying alloys, it predicts their numerical SFE values directly.

To accomplish this, QSVR introduces the **ε-insensitive loss function** through the **epsilon (ε) hyperparameter**, which defines a tolerance region around the true value where small prediction errors are ignored.

This allows the model to focus on significant prediction errors while remaining less sensitive to minor deviations.

---

# Overall Summary

The primary distinction between the two algorithms lies in the type of prediction they perform:

- **QSVC** performs **binary classification**, predicting whether an alloy's stacking fault energy falls above or below a specified threshold.
- **QSVR** performs **continuous regression**, estimating the actual stacking fault energy of the alloy.

Despite this difference, both methods rely on the same quantum kernel framework and share nearly identical preprocessing, validation, and experiment management pipelines. The transition from QSVC to QSVR primarily involves changing the learning objective from classification to regression and introducing the **epsilon (ε)** hyperparameter that is unique to Support Vector Regression.