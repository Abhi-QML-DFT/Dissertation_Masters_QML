# Python Imports Explained for Classical Machine Learning (CML) and Quantum Machine Learning (QML)

---

# System Utilities

These are standard Python libraries used for interacting with the operating system and handling command-line operations.

## `sys`

Think of `sys` as Python's communication channel with the operating system.

It provides access to:

- Python version
- Command-line arguments
- System paths
- Memory-related information

### Example

```python
import sys
print(sys.version)
```

**Output**

```text
Python 3.11.4
```

### Remark

Not particularly useful for this project, but commonly imported.

---

## `getopt`

Used for reading command-line arguments.

### Example

```bash
python train.py --epochs 50
```

`getopt` helps interpret options such as:

- `--epochs`
- `--batch_size`
- `--learning_rate`

### Remark

Not required for reproducing this research paper.

---

## `os`

`os` stands for **Operating System**.

It allows Python to interact with files and directories.

### Common Examples

```python
os.getcwd()
```

Returns the current working directory.

```python
os.listdir()
```

Lists files in the current directory.

```python
os.mkdir("Results")
```

Creates a new folder.

### Why it's useful

Machine learning projects constantly work with:

- datasets
- saved models
- plots
- output files

Therefore, `os` is almost always imported.

---

# Numerical Computing

## NumPy

```python
import numpy as np
```

NumPy is the fundamental library for numerical computation in Python.

It provides:

- Arrays
- Matrix operations
- Mathematical functions
- Linear algebra

Almost every ML project depends on NumPy.

---

# Data Handling

## Pandas

```python
import pandas as pd
```

Pandas is used for working with datasets.

Typical tasks include:

- Reading CSV files
- Cleaning data
- Selecting columns
- Filtering rows
- Handling missing values

Most materials science datasets are loaded using Pandas.

---

# Visualization

## Matplotlib

```python
import matplotlib.pyplot as plt
```

Matplotlib is the standard plotting library.

It is used to create:

- Line plots
- Scatter plots
- Histograms
- Training curves
- Prediction comparisons

Plots are essential for analysing machine learning performance.

---

# Classical Machine Learning (Scikit-Learn)

---

## RepeatedKFold

```python
from sklearn.model_selection import RepeatedKFold
```

Scikit-learn is one of the most widely used machine learning libraries in Python.

### What is K-Fold Cross Validation?

Normally, data is divided into:

- Training set
- Testing set

K-Fold instead divides the dataset into **K smaller parts**.

Example:

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Each fold becomes the testing set once while the remaining folds are used for training.

---

### What is RepeatedKFold?

RepeatedKFold repeats the entire K-Fold process multiple times using different random splits.

This produces a **more reliable estimate** of model performance.

### Why use it?

Materials science datasets are usually small.

A single train-test split may produce misleading results.

RepeatedKFold reduces this randomness and provides a better estimate of how well the model generalizes.

---

## Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score
```

These metrics evaluate how well the model performs.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

Lower values indicate better predictions.

---

### R² Score

Measures how much of the variation in the data is explained by the model.

Typical interpretation:

- **1.0** → Perfect prediction
- **0.9** → Excellent
- **0.5** → Moderate
- **0** → No improvement over simply predicting the average

---

## Data Scaling

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

Before training, features are often scaled because many machine learning algorithms perform better when all variables are on similar numerical scales.

---

### MinMaxScaler

Transforms every feature into a fixed range.

Usually:

```text
0 → 1
```

Useful when every feature should remain bounded within the same interval.

---

### StandardScaler

Transforms each feature so that:

```text
Mean = 0
Standard Deviation = 1
```

Many algorithms—including Support Vector Machines (SVMs)—perform particularly well with standardized data.

### Important Note

The original code comments:

```python
# StandardScaler is sensitive to outlier
```

This means extremely large or small values can heavily influence:

- the mean
- the standard deviation

As a result, the scaling may become less representative of the majority of the data.

---

# Quantum Machine Learning (QML)

---

## ZZFeatureMap

```python
from qiskit.circuit.library import ZZFeatureMap
```

One of the biggest questions in Quantum Machine Learning is:

> **How do we convert ordinary numerical data into a quantum circuit?**

A **Feature Map** performs this conversion.

Instead of leaving data as ordinary numbers,

```text
CSV Data
↓

Quantum Circuit
```

The data is encoded into qubits.

---

### Why ZZFeatureMap?

`ZZFeatureMap` is a predefined encoding circuit provided by Qiskit.

It uses:

- Single-qubit rotations
- Two-qubit entangling interactions

Because it captures interactions between features, it is widely used in:

- Quantum Kernel methods
- QSVC
- QSVR

---

## QSVC

```python
from qiskit_machine_learning.algorithms import QSVC
```

QSVC stands for:

**Quantum Support Vector Classifier**

It is the quantum version of a classical Support Vector Machine.

The main difference is:

### Classical SVM

Uses a classical kernel such as:

- Linear
- Polynomial
- Radial Basis Function (RBF)

### QSVC

Uses a **Quantum Kernel** computed from quantum circuits.

The rest of the SVM optimization remains largely unchanged.

---

## FidelityQuantumKernel

```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
```

This component computes the **quantum kernel**.

### What is a Kernel?

A kernel measures how similar two data points are.

---

### Classical Machine Learning

Similarity is computed using mathematical functions, such as:

- Linear kernel
- Polynomial kernel
- RBF kernel

---

### Quantum Machine Learning

Instead of mathematical formulas, similarity is computed using **quantum state fidelity**.

The procedure is:

```text
Classical Data
        ↓
Encode into Quantum States
        ↓
Compare the Quantum States
        ↓
Compute Fidelity
        ↓
Generate Quantum Kernel Matrix
        ↓
Pass Kernel Matrix to QSVC
```

The more similar two quantum states are, the larger their fidelity.

QSVC then uses this quantum kernel to perform classification.

---

# Overall Pipeline

The complete workflow used in this paper is:

```text
Dataset (CSV)
      ↓
Pandas
      ↓
Feature Scaling
      ↓
Repeated K-Fold Validation
      ↓
ZZFeatureMap
      ↓
FidelityQuantumKernel
      ↓
QSVC
      ↓
Predictions
      ↓
Evaluate using MSE / R² (or classification metrics where appropriate)
```

---

# Key Takeaways

- **NumPy** → Numerical computations
- **Pandas** → Dataset handling
- **Matplotlib** → Visualization
- **RepeatedKFold** → Reliable cross-validation for small datasets
- **MSE / R²** → Evaluate model performance
- **MinMaxScaler / StandardScaler** → Normalize features before training
- **ZZFeatureMap** → Encodes classical data into quantum circuits
- **FidelityQuantumKernel** → Measures similarity between quantum states
- **QSVC** → Performs classification using a quantum kernel




# Global Variables and Experiment Configuration

---

## Output Folder

```python
root_folder = 'QSVC'
```

Creates a folder named **QSVC** where the notebook stores all generated outputs, such as:

- Prediction results
- Graphs and plots
- Performance metrics
- Saved models (if any)
- Other experiment outputs

Keeping everything inside one folder makes the experiment organized and reproducible.

---

# Global Variables

The following variables are defined globally.

This means they can be accessed anywhere in the notebook without redefining them.

For example:

```python
NUM_FEATURES
NUM_QUBITS
N_REPEATS
```

can be used inside any function or code cell.

---

# Random Seed

```python
np.random.seed(42)
```

A **random seed** fixes Python's random number generator.

Think of it as telling Python:

> "Whenever you generate random numbers, always begin from exactly the same starting point."

Without a random seed:

```text
Run 1
↓

Different random numbers

↓

Different train/test split

↓

Different accuracy
```

With

```python
np.random.seed(42)
```

every execution becomes:

```text
42
↓

Same random numbers

↓

Same train/test split

↓

Same results
```

### Why is this important?

Scientific research requires **reproducibility**.

Anyone running the notebook should obtain identical experimental results.

---

# Number of Features

```python
NUM_FEATURES = 3
```

A **feature** is an input variable used by the machine learning model.

This notebook uses **three material descriptors**:

1. Electronegativity
2. Bulk Modulus
3. Atomic Volume

Therefore,

```text
Input Features

Feature 1 → Electronegativity

Feature 2 → Bulk Modulus

Feature 3 → Atomic Volume
```

---

# Number of Qubits

```python
NUM_QUBITS = NUM_FEATURES
```

This line connects the **classical machine learning problem** with the **quantum circuit**.

The feature map (ZZFeatureMap) encodes classical features into quantum states.

Therefore,

```text
3 Features

↓

3 Qubits
```

or equivalently,

```text
Feature 1 → Qubit 1

Feature 2 → Qubit 2

Feature 3 → Qubit 3
```

### Why?

Each feature requires one qubit for encoding.

If there were:

- 5 features → 5 qubits
- 8 features → 8 qubits

The number of qubits scales with the number of input features.

---

# Number of Targets

```python
NUM_TARGETS = 1
```

This specifies the number of output variables.

The model receives:

```text
Input

3 Features
```

and predicts

```text
1 Target

Stacking Fault Energy (SFE)
```

---

# Feature Map Repetition List

```python
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
```

A **Feature Map** converts classical numerical data into a quantum circuit.

The parameter **reps** controls how many times the encoding circuit is repeated.

Increasing the number of repetitions generally creates a more expressive quantum feature space, but also increases circuit depth and computational cost.

The authors are not choosing a single value.

Instead, they test multiple possibilities:

```text
reps = 1

reps = 2

reps = 3

reps = 4

reps = 5
```

This process is called a **hyperparameter sweep**.

The objective is to determine which circuit depth provides the best classification performance.

---

# Regularization Parameter List

```python
REGU_PARA_LIST = [0.1, 1, 10, 100]
```

The regularization parameter (**C**) controls how flexible the Support Vector Machine is.

Imagine trying to separate two classes.

A highly flexible model may produce a very complicated decision boundary that perfectly fits the training data but performs poorly on unseen data (overfitting).

A more regularized model prefers a simpler decision boundary that usually generalizes better.

The notebook evaluates four different values:

```text
C = 0.1

C = 1

C = 10

C = 100
```

Again, this is a **hyperparameter search**.

The authors want to determine:

> **Which regularization value produces the best classification performance?**

---

# Entanglement List

```python
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']
```

Entanglement specifies **which qubits are allowed to interact** inside the quantum feature map.

Different interaction patterns generate different quantum feature spaces.

---

## Linear Entanglement

```text
Q0 —— Q1 —— Q2
```

Each qubit interacts only with its nearest neighbours.

Advantages:

- Simple
- Shallower quantum circuit
- Lower computational cost

---

## Full Entanglement

```text
      Q0
     /  \
    /    \
   Q1----Q2
```

Every qubit interacts with every other qubit.

Advantages:

- Most expressive feature map
- Captures more complex relationships

Disadvantages:

- More quantum gates
- Higher computational cost
- Greater circuit depth

---

## Circular Entanglement

```text
Q0 —— Q1
|       |
Q2 -----
```

The qubits form a closed ring.

Unlike linear entanglement, the first and last qubits also interact.

This provides a balance between expressiveness and circuit complexity.

---

### Why test different entanglement structures?

The entanglement topology directly influences the quantum kernel.

Different datasets may perform better with different interaction patterns.

The authors therefore compare all three configurations to determine which produces the highest classification accuracy.

---

# Number of Repeated Cross-Validation Runs

```python
N_REPEATS = 10
```

The notebook performs repeated cross-validation **ten times**.

Each repetition creates a different random train-test split.

Instead of trusting the result from only one split, the experiment repeats the entire evaluation multiple times.

This provides:

- More reliable accuracy estimates
- Reduced influence of randomness
- Better assessment of model generalization

Repeated cross-validation is especially valuable for **small materials science datasets**, where performance can vary significantly depending on how the data is divided.

---

# Classification Threshold

```python
CLASSIFIER_THRESHOLD = 19
```

Originally,

Stacking Fault Energy (SFE) is a **continuous numerical value**.

Example:

```text
12.3

18.7

24.5

30.1
```

However, QSVC is a **classification algorithm**, not a regression algorithm.

Therefore, the continuous SFE values are converted into two categories.

The notebook defines:

```text
SFE < 19

↓

Class 0

(Low SFE)
```

and

```text
SFE ≥ 19

↓

Class 1

(High SFE)
```

Instead of predicting the exact stacking fault energy, QSVC predicts whether a material belongs to the:

- Low-SFE class
- High-SFE class

This conversion transforms the original **regression problem** into a **binary classification problem**, making it suitable for QSVC.


# Preparing One Cross-Validation Dataset Split

This function prepares **one train-test split** for cross-validation.

Its responsibilities are to:

- Separate the dataset into training and testing sets
- Remove non-numerical columns (element names)
- Scale the numerical features
- Return everything required for model training and evaluation

---

# Function Arguments

```python
prepare_dataset_k_fold(X, y, train_indices, test_indices)
```

The function receives four arguments:

- `X` → Feature matrix
- `y` → Target labels
- `train_indices` → Indices selected for training
- `test_indices` → Indices selected for testing

---

# Feature Matrix (`X`)

`X` contains the input features for every material.

Example:

| Element | Electronegativity | Bulk Modulus | Volume |
|----------|------------------:|-------------:|-------:|
| Mg-Al | 1.42 | 37 | 14.2 |
| Mg-Zn | 1.65 | 41 | 13.8 |
| Mg-Y | 1.22 | 28 | 18.1 |

Each row represents one material.

Each column (except the first) represents one numerical feature used by the machine learning model.

---

# Target Labels (`y`)

`y` contains the answers the model is trying to learn.

For this QSVC notebook, the regression problem has already been converted into binary classification.

Example:

```text
0
1
0
1
```

where

```text
0 → Low Stacking Fault Energy

1 → High Stacking Fault Energy
```

---

# Training Indices

During Repeated K-Fold Cross Validation, the algorithm does **not** copy data.

Instead, it only stores the row numbers belonging to the training set.

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

Internally this becomes

```python
train_indices = [0, 2, 3, 5, 6, 8]
```

These numbers simply indicate which rows belong to the training set.

---

# Testing Indices

Likewise,

```python
test_indices = [1, 4, 7]
```

means that rows

```text
1
4
7
```

will be used as the testing set.

---

# Splitting the Dataset

```python
X_train_raw, X_test_raw = X[train_indices], X[test_indices]

y_train, y_test = y[train_indices], y[test_indices]
```

This is the actual train-test split.

The feature matrix and labels are divided into:

- Training features
- Testing features
- Training labels
- Testing labels

At this stage, **no preprocessing has been performed**.

The datasets still contain every original column.

---

# Extracting Element Names

```python
element_test = X_test_raw[:, 0]

element_train = X_train_raw[:, 0]
```

The notation

```python
[:, 0]
```

means:

> Select **every row**, but only the **first column**.

Since the first column contains material names,

Example:

```text
Mg-Al

Mg-Zn

Mg-Y
```

only the element names are extracted.

These names are useful later when displaying predictions or analysing results.

They are **not** used as machine learning features.

---

# Removing the Element Names

```python
X_train = X_train_raw[:, 1:]

X_test = X_test_raw[:, 1:]
```

The notation

```python
[:, 1:]
```

means

> Keep every row, but start from column 1 onward.

This removes the first column containing strings.

Before:

| Element | Electronegativity | Bulk Modulus | Volume |
|----------|------------------:|-------------:|-------:|
| Mg-Al | 1.42 | 37 | 14.2 |

After:

| Electronegativity | Bulk Modulus | Volume |
|------------------:|-------------:|-------:|
| 1.42 | 37 | 14.2 |

Now the dataset contains only numerical values suitable for machine learning.

---

# Combining Training and Testing Features

```python
full_X = np.vstack([X_train, X_test])
```

`np.vstack()` stands for **vertical stack**.

Example:

Training:

```text
1 2 3
4 5 6
```

Testing:

```text
7 8 9
```

After

```python
np.vstack(...)
```

the result becomes

```text
1 2 3
4 5 6
7 8 9
```

The two datasets are temporarily combined.

This combined dataset is only used for computing the scaling transformation.

---

# Creating the Scaler

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
```

This creates a **MinMaxScaler** object.

At this point,

**nothing has been scaled yet.**

Think of it like buying a ruler.

The ruler exists, but you have not measured anything.

---

### Why Scale the Data?

Suppose the features have very different numerical ranges.

| Feature | Typical Values |
|----------|---------------:|
| Electronegativity | 1–4 |
| Bulk Modulus | 20–250 |
| Volume | 10–30 |

Without scaling,

Bulk Modulus has values around 200,

while Electronegativity has values around 2.

Large numerical ranges can dominate many machine learning algorithms.

Scaling places every feature on the same numerical scale.

In this notebook,

all features are transformed into the interval

```text
-1 → 1
```

---

# Learning the Scaling Parameters

```python
scaler.fit(full_X)
```

The `fit()` method **does not modify the data.**

Instead, it learns:

- minimum value of each feature
- maximum value of each feature

These values are stored inside the scaler.

Think of this as calibration.

The scaler is learning **how** future data should be transformed.

---

# Applying the Scaling

```python
X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

Now the learned transformation is applied.

First,

the training features are scaled.

Then,

the **same** transformation is applied to the testing features.

Using the **same scaler** is essential.

Otherwise,

training and testing data would exist in different numerical spaces, making the model evaluation inconsistent.

---

# An Important Research Observation

Notice that the notebook performs

```python
scaler.fit(full_X)
```

where

```text
full_X

=

Training Data

+

Testing Data
```

This means the scaler learns the minimum and maximum values using **both** training and testing samples.

For the purpose of **reproducing the published work**, this is perfectly acceptable because our goal is to match the authors' implementation exactly.

However, from a modern machine learning perspective, this introduces a small amount of **data leakage**, since information from the testing set influences the scaling process.

A more rigorous approach would be:

```python
scaler.fit(X_train)

↓

X_train_scaled = scaler.transform(X_train)

↓

X_test_scaled = scaler.transform(X_test)
```

Here, the scaler learns only from the training data within each cross-validation fold.

This ensures that the testing set remains completely unseen until evaluation.

---

# Research Insight

One of the most important transitions in research is moving from:

> **"Can I reproduce the authors' results?"**

to

> **"Can I identify small methodological choices that could be improved?"**

Recognizing design decisions such as how feature scaling is performed is an essential step toward critically evaluating research papers and eventually proposing improved methodologies of your own.


# Building the Quantum Support Vector Classifier (QSVC)

This function constructs a **Quantum Support Vector Classifier (QSVC)** using a given set of hyperparameters.

Its purpose is to build a model that is **ready for training**, but **does not train it yet**.

The function takes four inputs:

```python
(feature_dimension, C, reps, entangle)
```

and returns **one configured QSVC model**.

---

# Function Inputs

## `feature_dimension`

This specifies the number of input features.

Earlier in the notebook we defined:

```python
NUM_FEATURES = 3
```

Therefore,

```python
feature_dimension = 3
```

The quantum feature map will create one encoded quantum input for each classical feature.

In this project:

```text
Feature 1 → Electronegativity

Feature 2 → Bulk Modulus

Feature 3 → Atomic Volume
```

which becomes

```text
Qubit 1

Qubit 2

Qubit 3
```

---

## `C`

`C` is the **regularization parameter** inherited from the classical Support Vector Machine (SVM).

It controls how much the classifier penalizes classification errors during training.

### Small C

```text
High Regularization

↓

Simpler Decision Boundary

↓

Allows Some Classification Errors

↓

Better Generalization
```

A small value of `C` makes the model more tolerant of mistakes in the training data, reducing the risk of overfitting.

---

### Large C

```text
Low Regularization

↓

More Complex Decision Boundary

↓

Attempts to Correct Every Error

↓

Higher Risk of Overfitting
```

A large value of `C` forces the model to classify the training data as accurately as possible, even if it creates a complicated boundary that may not generalize well.

---

### Relationship Between Regularization and `C`

The notebook comments state:

> **Regularization is inversely proportional to `C`.**

This means:

| C Value | Regularization | Model Complexity |
|---------:|---------------:|-----------------:|
| Small | High | Simpler |
| Large | Low | More Complex |

The notebook evaluates four values:

```python
C = 0.1

C = 1

C = 10

C = 100
```

The authors perform a **hyperparameter sweep** because the optimal value is not known beforehand.

---

# `reps`

This parameter controls the number of repetitions of the quantum feature map.

It was introduced earlier as:

```python
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
```

Increasing `reps` increases the depth of the quantum circuit.

A larger number of repetitions generally allows the circuit to represent more complex relationships between features, but it also increases computational cost and circuit complexity.

---

# `entangle`

This parameter specifies the entanglement topology used by the quantum feature map.

Possible values are:

```python
'linear'

'full'

'circular'
```

Each topology determines how the qubits are allowed to interact.

Different interaction patterns produce different quantum feature spaces, which may influence the classifier's performance.

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

This line constructs the **ZZFeatureMap**.

A feature map performs one of the most important tasks in Quantum Machine Learning:

> **It converts classical numerical data into a quantum circuit.**

Instead of working directly with numbers,

```text
Classical Features

↓

Quantum States
```

The data is encoded onto qubits.

---

## Why Use `ZZFeatureMap`?

The authors selected `ZZFeatureMap` because it introduces **entanglement** between qubits.

Without entanglement:

```text
Feature 1

Feature 2

Feature 3

↓

Encoded Independently
```

With entanglement:

```text
Feature 1

↕
Feature 2

↕
Feature 3
```

The encoded quantum state can now capture **relationships between different features**, rather than treating each feature independently.

This is one of the primary motivations for using quantum feature maps in kernel-based quantum machine learning.

---

## `insert_barriers=True`

```python
insert_barriers=True
```

A **barrier** is a visual separator inserted into the quantum circuit.

It has **no effect on the computation itself**.

Its purpose is simply to make the circuit diagram easier to read by separating different stages of the circuit.

For example,

instead of displaying one long sequence of gates,

the barriers visually divide the circuit into logical sections.

This improves readability and debugging but does not change the quantum computation.

---

# Creating the Quantum Kernel

```python
kernel = FidelityQuantumKernel(feature_map=feature_map)
```

This line constructs the **quantum kernel**.

A kernel measures **how similar two data samples are**.

---

## Example

Suppose we have two alloys:

```text
Mg-Al

Mg-Zn
```

The quantum kernel asks:

> **"How similar are these two materials?"**

The process is:

```text
Mg-Al

↓

Encode into Quantum State
```

and

```text
Mg-Zn

↓

Encode into Quantum State
```

The kernel then compares the two quantum states by computing their **quantum state fidelity**.

---

### Fidelity

Fidelity measures the similarity between two quantum states.

```text
High Fidelity

↓

Very Similar Quantum States

↓

Likely Similar Materials
```

```text
Low Fidelity

↓

Very Different Quantum States

↓

Likely Different Materials
```

Instead of using a classical mathematical function (such as a linear or RBF kernel), the similarity is derived directly from quantum circuits.

This similarity matrix is called the **Quantum Kernel Matrix**.

---

# Constructing the QSVC Model

```python
qsvc = QSVC(
    C=C,
    quantum_kernel=kernel
)
```

The quantum kernel is now passed into the Quantum Support Vector Classifier.

An important observation is that **QSVC is still fundamentally a Support Vector Machine**.

The optimization procedure remains classical.

The **quantum** aspect lies in **how the similarity between samples is computed**.

The workflow becomes:

```text
Classical Data

↓

Quantum Feature Map

↓

Quantum Kernel Matrix

↓

Classical Support Vector Machine

↓

Classification
```

The quantum computer (or simulator) is responsible only for generating the kernel matrix.

The Support Vector Machine then uses this matrix to construct the decision boundary.

---

# Returning the Model

```python
return qsvc
```

Finally, the function returns the fully configured QSVC model.

At this stage:

- The feature map has been created.
- The quantum kernel has been constructed.
- The classifier has been configured.

However,

**the model has not yet learned anything from the data.**

No training has occurred.

Training only begins later when a statement such as

```python
qsvc.fit(X_train, y_train)
```

is executed.

Until then, this function simply builds a QSVC that is ready to be trained.

---

# Overall Workflow

The complete process performed by this function is:

```text
Input Hyperparameters

(feature_dimension, C, reps, entangle)

↓

Construct ZZFeatureMap

↓

Encode Classical Features into Quantum Circuits

↓

Construct FidelityQuantumKernel

↓

Measure Similarity Between Quantum States

↓

Pass Quantum Kernel into QSVC

↓

Return an Untrained QSVC Model
```

---

# Key Takeaways

- **`feature_dimension`** → Number of input features (and qubits).
- **`C`** → Controls the regularization strength of the Support Vector Machine.
- **`reps`** → Determines how many times the quantum feature map is repeated.
- **`entangle`** → Specifies how qubits interact within the feature map.
- **`ZZFeatureMap`** → Encodes classical data into entangled quantum states.
- **`FidelityQuantumKernel`** → Computes similarity between encoded quantum states.
- **`QSVC`** → Uses the quantum kernel within the classical SVM framework.
- **`return qsvc`** → Returns a fully configured model that is ready for training but has not yet been fitted to the data.


# Training the Quantum Support Vector Classifier (QSVC)

This function is where the Quantum Support Vector Classifier (QSVC) changes from an **empty, untrained model** into a **trained classifier**.

Unlike the previous function, which only *constructed* the model, this function actually teaches the model using the training data.

---

# Function Definition

```python
def train_qsvc(qsvc, X_train, y_train, X_test):
```

## Inputs

The function takes four arguments:

- `qsvc` → The Quantum Support Vector Classifier model
- `X_train` → Training feature matrix
- `y_train` → Training labels
- `X_test` → Testing feature matrix

---

## Output

The function returns:

- Predictions for the training data
- Predictions for the testing data

Notice that **the trained model itself is not returned**.

Instead, the function returns the model's predictions.

---

# The QSVC Model (`qsvc`)

```python
qsvc
```

This is the model that was created in the previous function.

At this stage, the model already knows:

- Which **quantum feature map** to use (`ZZFeatureMap`)
- Which **quantum kernel** to use (`FidelityQuantumKernel`)
- Which **regularization parameter (`C`)** to use

However, it **does not yet know anything about the dataset**.

Think of it as a brand-new student who has received a textbook but has not yet attended any lectures.

---

# Training Features (`X_train`)

`X_train` contains the numerical features used for learning.

Example:

| Electronegativity | Bulk Modulus | Volume |
|------------------:|-------------:|-------:|
| -0.30 | 0.65 | -0.10 |
| 0.25 | -0.18 | 0.41 |

Notice that the values are no longer in their original units.

Earlier in the notebook, they were scaled using **MinMaxScaler** into the range:

```text
-1 → 1
```

Scaling ensures that all features contribute more equally during training.

---

# Training Labels (`y_train`)

These are the correct answers that the model must learn.

Example:

```text
1
0
1
0
1
```

or equivalently,

```text
High SFE

Low SFE

High SFE

Low SFE

High SFE
```

The purpose of training is to discover the relationship between:

```text
Input Features

↓

Correct Class
```

---

# Testing Features (`X_test`)

`X_test` contains samples that the model has **never seen before**.

These samples are **not** used during training.

Their purpose is to evaluate whether the model can correctly classify completely unseen materials.

Testing on unseen data provides a better estimate of how well the model will perform in real-world applications.

---

# Training the Model

```python
qsvc.fit(X_train, np.concatenate(y_train))
```

This is the most important line in the entire function.

The `.fit()` method means:

> **Learn from the training data.**

Before this line executes:

```text
QSVC

↓

No knowledge

↓

Cannot classify materials
```

After this line executes:

```text
QSVC

↓

Learns from Training Data

↓

Ready to Classify New Materials
```

This is the point where the model is actually trained.

---

# What Happens Internally During `.fit()`?

Although this appears to be a single line of code, a considerable amount of computation takes place behind the scenes.

The process is approximately:

```text
Training Features

↓

Encode into Quantum Circuits
(using ZZFeatureMap)

↓

Generate Quantum States

↓

Compute Quantum State Fidelities

↓

Construct Quantum Kernel Matrix

↓

Pass Kernel Matrix to Classical SVM

↓

Solve Optimization Problem

↓

Store Learned Decision Boundary
```

An important point is that **the quantum computer (or simulator) is not directly performing the classification**.

Instead:

- The quantum circuit computes the **quantum kernel matrix**, measuring similarities between samples.
- A classical Support Vector Machine then uses this kernel matrix to solve its optimization problem and learn the decision boundary.

Thus, QSVC is a **hybrid quantum-classical algorithm**.

---

# Why Use `np.concatenate(y_train)`?

```python
np.concatenate(y_train)
```

This operation adjusts the shape of the training labels.

Machine learning models generally expect labels in the form of a **one-dimensional array**.

For example:

Instead of

```text
[[1]
 [0]
 [1]
 [0]]
```

the labels become

```text
[1 0 1 0]
```

This transformation ensures compatibility with the QSVC implementation.

---

# Making Predictions

```python
qsvc.predict(X_train)
```

Once the model has been trained, it can make predictions.

For every training sample, the classifier predicts:

```text
Low SFE

or

High SFE
```

These predictions are often used to evaluate how well the model learned the training data.

---

```python
qsvc.predict(X_test)
```

The same trained model is then applied to the testing data.

Since these samples were **never used during training**, the predictions provide an unbiased estimate of the model's ability to generalize.

This is the most important performance measure in machine learning.

---

# Returning the Predictions

```python
return qsvc.predict(X_train), qsvc.predict(X_test)
```

The function returns two arrays:

1. Predictions for the training set.
2. Predictions for the testing set.

For example,

Training predictions:

```text
[1 0 1 1 0]
```

Testing predictions:

```text
[0 1 0]
```

These predicted labels can later be compared with the true labels to compute performance metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Why Doesn't the Function Return the Model?

You might wonder why the function returns only the predictions instead of the trained model.

The reason is that the model (`qsvc`) is **modified in place**.

When `.fit()` is called, the original `qsvc` object itself becomes trained.

There is no need to create or return a new model object.

Instead, the notebook immediately uses the trained model to generate predictions and returns those predictions for later evaluation.

---

# Behind the Scenes

Although this function contains only two executable lines,

```python
qsvc.fit(...)

qsvc.predict(...)
```

it hides almost all of the computational work performed by QSVC.

When `.fit()` is called:

1. The **ZZFeatureMap** encodes every training sample into a quantum state.
2. The **FidelityQuantumKernel** computes the similarity (fidelity) between every pair of encoded quantum states.
3. These similarities form the **Quantum Kernel Matrix**.
4. The kernel matrix is passed to a **classical Support Vector Machine**, which solves the optimization problem and learns the optimal decision boundary.

When `.predict()` is called:

1. New samples are encoded into quantum states.
2. Their similarities to the learned support vectors are computed using the quantum kernel.
3. The trained SVM uses these similarities to determine the predicted class.

This is why QSVC is described as a **hybrid quantum-classical algorithm**: the quantum component computes similarities, while the classical component performs the optimization and classification.

---

# Overall Workflow

```text
Training Data
      ↓
Scale Features
      ↓
ZZFeatureMap
      ↓
Encode into Quantum States
      ↓
Compute State Fidelities
      ↓
Generate Quantum Kernel Matrix
      ↓
Classical SVM Optimization
      ↓
Trained QSVC
      ↓
Predict Training Samples
      ↓
Predict Testing Samples
      ↓
Return Predictions
```

---

# Key Takeaways

- **`qsvc`** → An untrained QSVC model configured with a feature map, quantum kernel, and regularization parameter.
- **`X_train`** → Scaled training features used to teach the model.
- **`y_train`** → Correct class labels that guide the learning process.
- **`X_test`** → Unseen data reserved exclusively for evaluating model performance.
- **`.fit()`** → Encodes training data into quantum states, computes the quantum kernel matrix, and trains the classical SVM.
- **`np.concatenate(y_train)`** → Converts the training labels into the one-dimensional format expected by QSVC.
- **`.predict()`** → Uses the trained model to classify both training and unseen testing samples.
- **`return`** → Returns the predicted classes for both datasets, which are later used to evaluate the classifier's performance.


# Reading Command-Line Arguments

Unlike the previous functions, this function **does not perform any machine learning or quantum computation**.

Its purpose is purely **software engineering**.

It allows the user to control the QSVC experiment by providing options directly from the command line instead of modifying the source code.

In other words, it makes the program easier to use and automate.

---

# Function Definition

```python
def get_arguments(argvs):
```

## Purpose

This function reads the options supplied by the user when the program is executed from a terminal.

For example, suppose the notebook was converted into a standalone Python script named `QSVC.py`.

Instead of opening the notebook and manually changing variables, the user could simply run:

```bash
python QSVC.py -e linear -f 3 -r 10
```

This command tells the program:

- Run `QSVC.py`
- Use **linear entanglement**
- Use **3 feature map repetitions**
- Use a **regularization parameter (`C`) of 10**

Without this function, Python would have no idea what the options `-e`, `-f`, and `-r` mean.

---

# The `argvs` Parameter

```python
argvs
```

`argvs` stands for **Argument Vector**.

It contains everything typed after the program name.

Example:

```bash
python QSVC.py -e linear -f 3 -r 10
```

Internally, Python receives something similar to:

```python
[
    "-e",
    "linear",
    "-f",
    "3",
    "-r",
    "10"
]
```

The job of `get_arguments()` is to interpret this list and convert it into meaningful program settings.

---

# Creating Empty Variables

```python
_entangle = ''

_feature_map_reps = ''

_regu_para = ''
```

These variables act as **containers** waiting to receive values from the command line.

Initially:

```text
_entangle           = ""

_feature_map_reps   = ""

_regu_para          = ""
```

After reading

```bash
python QSVC.py -e linear -f 3 -r 10
```

they become:

```text
_entangle = "linear"

_feature_map_reps = "3"

_regu_para = "10"
```

These values can then be used later to configure the QSVC model.

---

# Error Handling with `try` and `except`

The function uses Python's

```python
try
```

and

```python
except
```

statements.

Their purpose is to prevent the program from crashing if the user provides invalid command-line arguments.

For example,

```bash
python QSVC.py -x abc
```

contains an unsupported option (`-x`).

Without error handling:

```text
Program crashes

↓

Python traceback

↓

Execution stops
```

With `try` and `except`:

```text
Invalid option detected

↓

Display helpful error message

↓

Exit gracefully
```

This makes the program much more user-friendly.

---

# Parsing the Command-Line Arguments

```python
opts, args = getopt.getopt(
    argvs,
    "h:e:f:r:",
    ["entangle=", "feature_map_reps=", "_regu_para="]
)
```

This is the core of the function.

Earlier in the notebook,

`getopt` was imported from Python's standard library.

Its job is to examine the command-line input and separate:

- recognised options
- remaining arguments

The result is stored in two variables:

```python
opts
```

contains all recognised command-line options.

```python
args
```

contains any additional arguments that were not interpreted as options.

---

# Understanding the Short Flags

The string

```python
"h:e:f:r:"
```

defines the available **short command-line flags**.

| Flag | Meaning |
|------|----------|
| `-h` | Help |
| `-e` | Entanglement type |
| `-f` | Feature map repetitions |
| `-r` | Regularization parameter (`C`) |

The colon (`:`) after each letter indicates that the option requires a value.

For example,

```bash
-e linear
```

means

```text
Option

↓

-e

↓

Value

↓

linear
```

Likewise,

```bash
-f 3
```

means

```text
Feature Map Repetitions = 3
```

---

# Long Options

The function also supports longer, more descriptive versions of the same options.

For example,

instead of

```bash
-e linear
```

the user could write

```bash
--entangle=linear
```

Similarly,

```bash
--feature_map_reps=3
```

or

```bash
--_regu_para=10
```

Long options improve readability, especially in shell scripts or automated workflows.

---

# Does This Affect the QSVC Algorithm?

**No.**

From a Quantum Machine Learning perspective,

this function changes **nothing** about the mathematics.

Whether the hyperparameters are:

- hard-coded directly into the notebook, or
- supplied through command-line arguments,

the QSVC algorithm performs exactly the same computations.

The quantum feature map,

the quantum kernel,

and the Support Vector Machine remain unchanged.

---

# Why Is This Useful?

Although it does not change the algorithm,

this function is extremely useful from a **research software engineering** perspective.

Imagine the authors wanted to compare dozens of different QSVC configurations.

Instead of opening the notebook and manually changing variables every time,

they could simply execute commands such as:

```bash
python QSVC.py -e linear -f 1 -r 0.1
```

```bash
python QSVC.py -e linear -f 2 -r 0.1
```

```bash
python QSVC.py -e full -f 5 -r 100
```

Each command launches a different experiment without modifying the source code.

---

# Why Researchers Do This

On Linux workstations or High-Performance Computing (HPC) clusters,

researchers often need to run hundreds of experiments.

Instead of manually launching each one,

they write shell scripts such as:

```bash
for reps in 1 2 3 4 5
do
    python QSVC.py -e linear -f $reps -r 10
done
```

or submit multiple jobs through an HPC scheduler.

This allows experiments to run automatically, sometimes overnight or across many computing nodes.

Automation greatly improves productivity and reduces the chance of human error.

---

# Research Insight

This function provides an interesting clue about the history of the code.

The current project is presented as a Jupyter Notebook, but the presence of command-line argument parsing suggests that the code was **originally developed as a standalone Python script**.

Researchers commonly write scripts for large-scale experimentation and later adapt them into notebooks for publication or easier demonstration.

This indicates that the authors likely performed many QSVC experiments systematically before preparing the notebook version.

---

# Overall Workflow

```text
User Runs Program

↓

python QSVC.py -e linear -f 3 -r 10

↓

argvs Receives the Arguments

↓

getopt Parses the Arguments

↓

Extract Hyperparameter Values

↓

Configure QSVC Experiment

↓

Run Training and Evaluation
```

---

# Key Takeaways

- **`get_arguments(argvs)`** → Reads command-line options provided by the user.
- **`argvs`** → Contains all arguments entered after the program name.
- **Empty variables** → Serve as placeholders for the parsed hyperparameter values.
- **`try` / `except`** → Prevent the program from crashing when invalid arguments are supplied.
- **`getopt.getopt()`** → Parses command-line options into recognised flags and values.
- **Short flags (`-e`, `-f`, `-r`)** → Specify the entanglement type, feature map repetitions, and regularization parameter.
- **Long options (`--entangle`, `--feature_map_reps`, `--_regu_para`)** → Provide more descriptive alternatives to the short flags.
- **No effect on QSVC mathematics** → The function only changes how hyperparameters are supplied, not how the algorithm works.
- **Research advantage** → Enables automated large-scale experiments on Linux systems and HPC clusters without repeatedly editing the source code.


# Creating Output Directories for the Experiment

This code cell is another example of **software engineering** and **research organization**.

It has **no effect on the Quantum Support Vector Classifier (QSVC) algorithm itself**.

Instead, its purpose is to prepare the folder structure where all experiment outputs will be stored.

If the previous `get_arguments()` function handled the **input** to the program, this cell handles the **output**.

---

# The Code

```python
if not os.path.exists(f'{root_folder}/result'):
    os.makedirs(f'{root_folder}/result')

if not os.path.exists(f'{root_folder}/logs'):
    os.makedirs(f'{root_folder}/logs')
```

This code checks whether the required folders already exist.

If they do not, Python automatically creates them.

---

# Understanding `os.path.exists()`

```python
os.path.exists(path)
```

This function checks whether a file or folder already exists.

It returns:

```text
True
```

if the path exists,

or

```text
False
```

if it does not.

Example:

```python
os.path.exists("QSVC/result")
```

Possible outcomes:

```text
True
```

→ The folder already exists.

or

```text
False
```

→ The folder does not exist.

---

# Understanding `os.makedirs()`

```python
os.makedirs(path)
```

This function creates a directory (folder).

For example,

```python
os.makedirs("QSVC/result")
```

creates the following structure:

```text
QSVC/
└── result/
```

Similarly,

```python
os.makedirs("QSVC/logs")
```

creates:

```text
QSVC/
└── logs/
```

If intermediate folders do not exist, `makedirs()` can create the entire directory tree automatically.

---

# Why Use an `if` Statement?

Imagine the folder already exists.

Without checking first,

```python
os.makedirs("QSVC/result")
```

would raise an error because Python cannot create a folder that already exists.

The `if` statement prevents this.

The logic becomes:

```text
Does the folder exist?

↓

Yes

↓

Do nothing

↓

Continue
```

or

```text
Does the folder exist?

↓

No

↓

Create the folder

↓

Continue
```

This makes the code safe to run multiple times.

---

# Workflow

The logic of this cell can be visualized as:

```text
Start Program
      │
      ▼
Is "QSVC/result" present?
      │
 ┌────┴────┐
 │         │
Yes       No
 │         │
Skip    Create Folder
 │
 ▼
Is "QSVC/logs" present?
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

---

# What Will These Folders Contain?

Although this cell does not specify their contents, later parts of the notebook will likely save different types of outputs.

## `result/`

This folder typically stores the scientific results of the experiment, such as:

- Prediction CSV files
- Accuracy tables
- Confusion matrices
- Evaluation metrics
- Figures and plots
- Saved models (if applicable)

Example:

```text
QSVC/
└── result/
    ├── predictions.csv
    ├── accuracy.csv
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## `logs/`

This folder stores information about how the experiment was executed.

Typical log files include:

- Training progress
- Parameter settings
- Execution timestamps
- Error messages
- Debugging information

Example:

```text
QSVC/
└── logs/
    ├── experiment_01.log
    ├── experiment_02.log
    └── run_information.txt
```

Logs are especially useful when experiments are run automatically or on remote servers, where you may not be watching the program execute.

---

# Why Is This Important in Research?

Although this cell contains **no machine learning** or **quantum computing**, it demonstrates **good computational research practice**.

Scientific experiments often generate a large number of output files.

For example:

```text
Predictions

Plots

Accuracy Tables

Kernel Matrices

Log Files

CSV Files
```

Without organization, these files quickly become difficult to manage.

Automatically creating a consistent folder structure ensures that:

- Results are stored in predictable locations.
- Files are less likely to be overwritten accidentally.
- Different experiments remain organized.
- Collaborators can easily understand where outputs are located.

---

# Why Is This Good for Reproducibility?

One of the goals of scientific software is **reproducibility**.

Suppose another researcher downloads the notebook.

Without this cell, they would first have to manually create the required folders before running the code.

If they forget,

the notebook may fail when it attempts to save output files.

With this code:

```text
Download Repository

↓

Run Notebook

↓

Folders Created Automatically

↓

Experiment Runs Successfully
```

The code becomes portable across different computers and operating systems.

---

# Research Insight

This small piece of code illustrates an important distinction:

- **The QSVC algorithm** determines *how* the model learns from data.
- **Research software engineering** determines *how* experiments are organized, executed, and reproduced.

A well-designed research project is not only about implementing sophisticated algorithms—it is also about making the code reliable, reusable, and easy for others to run.

As you progress in research, you'll find that good organization often saves more time than adding new features.

---

# Overall Workflow

```text
Program Starts
      ↓
Check if "QSVC/result" Exists
      ↓
Create if Missing
      ↓
Check if "QSVC/logs" Exists
      ↓
Create if Missing
      ↓
Continue with QSVC Experiment
      ↓
Save Results and Logs Automatically
```

---

# Key Takeaways

- **`os.path.exists()`** → Checks whether a file or directory already exists.
- **`os.makedirs()`** → Creates directories automatically.
- **`if` statement** → Prevents errors by avoiding attempts to recreate existing folders.
- **`result/`** → Intended for experiment outputs such as predictions, plots, and evaluation metrics.
- **`logs/`** → Intended for execution logs and debugging information.
- **No effect on QSVC** → This cell does not change the machine learning or quantum algorithm.
- **Research benefit** → Automatically organizing outputs improves portability, reproducibility, and experiment management across different computers and research environments.



# Loading and Preparing the Dataset

This section loads the dataset from a CSV file and prepares it for the Quantum Support Vector Classifier (QSVC).

Every machine learning project begins by loading data into memory before it can be processed, split, scaled, and used for training.

---

# Dataset Location

```python
dataset_name = "./Desktop/ABHI/Learning,Reproducing/qml_training-validation-data.csv"
```

This line stores the location (path) of the dataset in a variable called `dataset_name`.

Instead of writing the file path multiple times throughout the notebook, it is stored once in a variable.

Advantages:

- Easier to modify later
- Cleaner code
- Less chance of typing errors

---

# Reading the Dataset

```python
df = pd.read_csv(dataset_name)
```

`pd.read_csv()` is a Pandas function used to load data stored in a **CSV (Comma-Separated Values)** file.

The CSV file is read and stored as a **Pandas DataFrame** named `df`.

A **DataFrame** is one of Pandas' most important data structures.

It behaves like a spreadsheet or database table.

Example:

| Element | el_neg | B/GPa | Volume/A³ | SFE/mJm⁻² |
|----------|-------:|------:|----------:|----------:|
| Mg-Al | 1.42 | 37 | 14.2 | 18.5 |
| Mg-Zn | 1.65 | 41 | 13.8 | 22.7 |
| Mg-Y | 1.22 | 28 | 18.1 | 15.6 |

Each row represents one alloy.

Each column represents one measured property.

---

# Displaying the Dataset

Instead of

```python
print(df)
```

the notebook uses

```python
display(df)
```

### Why use `display()`?

Both commands show the dataset, but they present it differently.

### `print(df)`

Produces plain text output.

Example:

```text
Element el_neg B/GPa Volume
Mg-Al   1.42   37    14.2
Mg-Zn   1.65   41    13.8
```

---

### `display(df)`

Produces a properly formatted table similar to a spreadsheet.

Advantages:

- Better alignment
- Easier to read
- Interactive in Jupyter Notebook
- More suitable for inspecting datasets

For notebooks, `display()` is generally preferred.

---

# Selecting the Input Features (`X`)

```python
X = df[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']].values
```

Machine learning separates the dataset into:

- **Input features** (`X`)
- **Target variable** (`y`)

The selected input features are:

- Element
- Electronegativity (`el_neg`)
- Bulk Modulus (`B/GPa`)
- Atomic Volume (`Volume/A³`)

These are the variables the model will use to make predictions.

---

## Why Double Square Brackets?

Notice:

```python
[['Element', 'el_neg', 'B/GPa', 'Volume/A^3']]
```

Double square brackets indicate that **multiple columns** are being selected.

Example:

```python
df[['A', 'B', 'C']]
```

returns

```text
Column A

Column B

Column C
```

as a new DataFrame.

---

# Selecting the Target Variable (`y`)

```python
y = df['SFE/mJm^-2'].values
```

The target variable is:

```text
Stacking Fault Energy (SFE)
```

This is the quantity the model is trying to predict.

Unlike `X`, only **one column** is required.

Therefore, only a **single pair of square brackets** is used.

Example:

```python
df['SFE/mJm^-2']
```

returns a single Pandas Series.

---

# Machine Learning Interpretation

The model learns the relationship:

```text
Input

↓

Element

Electronegativity

Bulk Modulus

Atomic Volume

↓

Output

Stacking Fault Energy
```

or more simply,

```text
Electronegativity

Bulk Modulus

Volume

↓

Stacking Fault Energy
```

The Element column is retained initially for identification purposes, but as seen in earlier preprocessing steps, it is later removed before numerical training because machine learning algorithms cannot directly process text labels.

---

# Converting to NumPy Arrays

Notice the use of:

```python
.values
```

Both `X` and `y` end with:

```python
.values
```

This converts the Pandas objects into **NumPy arrays**.

Why?

Pandas is excellent for:

- Reading files
- Cleaning data
- Manipulating tables

However,

most machine learning libraries—including Scikit-learn and Qiskit Machine Learning—expect input as **NumPy arrays**.

Example:

Before:

```text
Pandas DataFrame
```

After:

```text
NumPy ndarray
```

The conversion makes the data compatible with the QSVC implementation.

---

# Understanding `.shape`

```python
df.shape
```

The `.shape` attribute reports the dimensions of the dataset.

It returns:

```text
(rows, columns)
```

In this notebook,

```python
df.shape
```

produces:

```text
(21, 5)
```

This means:

- **21 rows** (21 material samples)
- **5 columns** (Element, Electronegativity, Bulk Modulus, Volume, and Stacking Fault Energy)

Visualized as:

```text
           Columns

      1   2   3   4   5

Rows

1

2

3

...

21
```

Knowing the dataset size is an important first step before preprocessing and model training.

---

# Why Is This Important?

Loading the dataset correctly is the foundation of the entire machine learning pipeline.

Every later stage depends on these data:

```text
CSV File
      ↓
Pandas DataFrame
      ↓
Select Features (X)
      ↓
Select Target (y)
      ↓
Convert to NumPy Arrays
      ↓
Train/Test Split
      ↓
Scaling
      ↓
QSVC Training
```

If the data are loaded incorrectly, every subsequent step will also be incorrect.

---

# Research Insight

Although this section appears simple, it illustrates a standard workflow followed in nearly every machine learning project:

1. Load the dataset.
2. Inspect the data.
3. Separate input features from the target variable.
4. Convert the data into a format suitable for machine learning.

This workflow is common not only in Quantum Machine Learning but also in classical machine learning, deep learning, and many computational materials science applications.

---

# Overall Workflow

```text
CSV Dataset
      ↓
pd.read_csv()
      ↓
Pandas DataFrame (df)
      ↓
Display Dataset
      ↓
Select Input Features (X)
      ↓
Select Target Variable (y)
      ↓
Convert to NumPy Arrays
      ↓
Ready for Preprocessing and QSVC Training
```

---

# Key Takeaways

- **`dataset_name`** → Stores the location of the dataset file.
- **`pd.read_csv()`** → Loads the CSV file into a Pandas DataFrame.
- **`display(df)`** → Displays the dataset in a clean tabular format within Jupyter Notebook.
- **`X`** → Contains the input features (`Element`, `Electronegativity`, `Bulk Modulus`, and `Volume`).
- **`y`** → Contains the target variable (`Stacking Fault Energy`).
- **Double square brackets (`[[ ]]`)** → Select multiple columns.
- **Single square brackets (`[ ]`)** → Select a single column.
- **`.values`** → Converts Pandas objects into NumPy arrays compatible with machine learning libraries.
- **`.shape`** → Reports the dataset dimensions as `(rows, columns)`, which in this case is `(21, 5)`.



# Converting the Regression Target into Classification Labels

This cell converts the original **Stacking Fault Energy (SFE)** values into **two classes**.

Originally, the dataset contains **continuous numerical values** for SFE.

Example:

| Alloy | SFE (mJ/m²) |
|--------|------------:|
| Mg-Al | 16.8 |
| Mg-Zn | 24.5 |
| Mg-Y | 18.3 |
| Mg-Ca | 27.1 |

These values are suitable for a **regression** problem.

However, the QSVC algorithm used in this notebook is a **classifier**, not a regressor.

Therefore, the continuous SFE values must first be converted into discrete class labels.

---

# Classification Threshold

Earlier in the notebook, the following global variable was defined:

```python
CLASSIFIER_THRESHOLD = 19
```

This threshold divides all materials into two categories.

The classification rule is:

```text
SFE < 19

↓

Class 0 (Low SFE)
```

```text
SFE ≥ 19

↓

Class 1 (High SFE)
```

*(Note: Your comment says "greater than 19 is assigned as 0 and less than 19 as 1", but based on the earlier explanation in the notebook and the usual implementation, it is typically **below the threshold → 0** and **above/equal to the threshold → 1**. It's worth checking the actual code to confirm the direction.)*

---

# Example

Suppose the original SFE values are

| Material | SFE |
|-----------|----:|
| Mg-Al | 16 |
| Mg-Zn | 22 |
| Mg-Y | 18 |
| Mg-Ca | 25 |

After applying the threshold:

| Material | SFE | Class |
|-----------|----:|------:|
| Mg-Al | 16 | 0 |
| Mg-Zn | 22 | 1 |
| Mg-Y | 18 | 0 |
| Mg-Ca | 25 | 1 |

The regression targets

```text
16

22

18

25
```

become

```text
0

1

0

1
```

The QSVC now learns to distinguish between **Low-SFE** and **High-SFE** materials instead of predicting the exact SFE value.

---

# Why Perform This Conversion?

Originally, the problem is

```text
Input Features

↓

Predict Exact SFE
```

This is a **regression** task.

After thresholding:

```text
Input Features

↓

Low SFE

or

High SFE
```

This becomes a **binary classification** problem.

Since QSVC is designed for classification, this conversion is essential.

---

# Reshaping the Labels

After classification, the labels are reshaped using

```python
reshape(-1, 1)
```

Suppose the labels initially look like this:

```text
0
1
1
0
```

Internally, this is a one-dimensional array:

```python
[0, 1, 1, 0]
```

After

```python
reshape(-1, 1)
```

it becomes

```python
[
 [0],
 [1],
 [1],
 [0]
]
```

This is now a **column vector**.

---

## Why Use `-1`?

The `-1` tells NumPy:

> "Automatically determine the correct number of rows."

Since there are four labels,

```python
reshape(-1, 1)
```

becomes

```python
reshape(4, 1)
```

If there were 21 labels,

it would become

```python
reshape(21, 1)
```

The number of rows is determined automatically.

---

# Scaling the Labels

Later, the notebook applies

```python
fit_transform(...)
```

to these labels.

After this transformation,

the numerical range changes from

```text
0

1
```

to

```text
-1

1
```

For example,

Before scaling:

```text
0

1

1

0
```

After scaling:

```text
-1

1

1

-1
```

The labels are now represented symmetrically around zero.

---

# Why Scale the Labels?

Although feature scaling is very common, scaling class labels is less common in standard classification workflows.

In this notebook, the transformation appears to be part of the authors' preprocessing pipeline.

When reproducing published work, it is important to preserve these implementation details exactly, even if they are not strictly necessary for all classification algorithms.

---

# Overall Workflow

```text
Original Target Values

(Stacking Fault Energy)

      ↓

Apply Threshold (19 mJ/m²)

      ↓

Low SFE        High SFE

0              1

      ↓

Reshape into Column Vector

      ↓

Scale Labels

0,1

↓

-1,1

      ↓

Ready for QSVC Training
```

---

# Research Insight

This cell represents an important modeling decision.

The authors deliberately changed the scientific question from:

> **"What is the exact stacking fault energy?"**

to

> **"Does this alloy belong to the low-SFE or high-SFE class?"**

This simplifies the prediction task and allows the use of the Quantum Support Vector Classifier.

Such choices are common in materials informatics, where predicting categories (e.g., stable/unstable, brittle/ductile, low/high property values) can sometimes be more practical than predicting precise numerical values.

---

# Key Takeaways

- **`CLASSIFIER_THRESHOLD = 19`** → Divides the continuous SFE values into two classes.
- **Thresholding** → Converts a regression problem into a binary classification problem.
- **Class labels** → Represent Low-SFE and High-SFE materials.
- **`reshape(-1, 1)`** → Converts a one-dimensional array into a column vector while automatically determining the required number of rows.
- **`fit_transform()`** → Scales the class labels from the range **0–1** to **−1–1** as part of the notebook's preprocessing pipeline.
- **Purpose** → Prepare the target labels in the format expected by the subsequent QSVC training process.


# Creating the Repeated K-Fold Cross-Validation Strategy

```python
rkf = RepeatedKFold(
    n_splits=X.shape[0] // TEST_SIZE,
    n_repeats=N_REPEATS
)
```

This line creates the **cross-validation strategy** that will be used throughout the experiment.

An important point is that **this line does not train the model**.

Instead, it simply creates a **plan** describing how the dataset will be divided into training and testing sets.

Think of it as preparing the experiment before any learning begins.

---

# What is Cross-Validation?

Suppose you have a dataset containing **100 alloys**.

A simple approach would be:

```text
100 Samples

↓

80 Training Samples

20 Testing Samples

↓

Train Once

↓

Test Once

↓

Finished
```

Although this approach is straightforward, the results depend heavily on **which 20 samples** happened to be selected as the testing set.

A different split could produce noticeably different performance.

This becomes an even greater problem when working with **very small datasets**, such as the one used in this paper.

---

# Why Not Train Only Once?

This project contains only

```text
21 materials
```

If just one train-test split is used,

the measured accuracy may be strongly influenced by which samples happen to be in the testing set.

One fortunate split may produce excellent accuracy,

while another equally valid split may produce much lower accuracy.

Instead of trusting one random split,

machine learning repeatedly evaluates the model using many different train-test partitions.

This approach is called **Cross-Validation**.

---

# What is K-Fold Cross-Validation?

K-Fold Cross-Validation divides the dataset into **K equal parts**, called **folds**.

Example:

Suppose we have

```text
12 samples
```

and choose

```text
K = 4
```

The data are divided into

```text
Fold 1

Fold 2

Fold 3

Fold 4
```

The experiment proceeds as follows:

### First Iteration

```text
Test

Fold 1

Training

Fold 2

Fold 3

Fold 4
```

---

### Second Iteration

```text
Training

Fold 1

Test

Fold 2

Training

Fold 3

Fold 4
```

---

### Third Iteration

```text
Training

Fold 1

Fold 2

Test

Fold 3

Training

Fold 4
```

---

### Fourth Iteration

```text
Training

Fold 1

Fold 2

Fold 3

Test

Fold 4
```

Every sample eventually becomes part of the testing set exactly once.

The final performance is obtained by averaging the results across all folds.

---

# What is `RepeatedKFold`?

`RepeatedKFold` extends ordinary K-Fold Cross-Validation.

Instead of performing one complete K-Fold experiment,

it repeats the **entire process multiple times** using different random data partitions.

The workflow becomes:

```text
First K-Fold Experiment

↓

Average Accuracy

↓

Randomly Shuffle Data

↓

Second K-Fold Experiment

↓

Average Accuracy

↓

Repeat Again

↓

Compute Overall Average
```

This produces a much more reliable estimate of model performance.

---

# Understanding `n_splits`

```python
n_splits = X.shape[0] // TEST_SIZE
```

This determines **how many folds** the dataset will be divided into.

Earlier,

```python
X.shape
```

returned

```text
(21, 4)
```

meaning:

- 21 samples
- 4 feature columns

Therefore,

```python
X.shape[0]
```

returns

```text
21
```

Earlier in the notebook,

```python
TEST_SIZE = 1
```

Therefore,

```python
21 // 1
```

equals

```text
21
```

So,

```python
n_splits = 21
```

---

# What Does 21 Folds Mean?

Each fold contains exactly

```text
1 testing sample
```

while the remaining

```text
20 samples
```

are used for training.

Example:

### Iteration 1

```text
Test

Sample 1

Training

Samples 2–21
```

---

### Iteration 2

```text
Test

Sample 2

Training

All Remaining Samples
```

---

### ...

---

### Iteration 21

```text
Test

Sample 21

Training

Samples 1–20
```

Every material is tested exactly once.

---

# Leave-One-Out Cross-Validation (LOOCV)

When

```text
Number of Folds

=

Number of Samples
```

each fold contains exactly one testing sample.

This special case is called

**Leave-One-Out Cross-Validation (LOOCV).**

The workflow is

```text
21 Samples

↓

Leave One Sample Out

↓

Train on Remaining 20

↓

Test on Left-Out Sample

↓

Repeat 21 Times
```

Every sample receives its own independent evaluation.

---

# Why Use LOOCV?

Small datasets are common in computational materials science.

With only

```text
21 materials
```

discarding a large testing set would waste valuable training data.

LOOCV allows the model to train on

```text
20 out of 21 samples
```

during every experiment.

This maximizes the amount of information available for learning while still providing an independent test for every material.

For small datasets, LOOCV is considered one of the most rigorous evaluation strategies.

---

# Understanding `n_repeats`

```python
n_repeats = N_REPEATS
```

Earlier in the notebook,

```python
N_REPEATS = 10
```

(or `1`, depending on the current configuration).

This parameter specifies how many times the **entire LOOCV procedure** should be repeated.

If

```python
N_REPEATS = 10
```

the workflow becomes:

```text
LOOCV

↓

21 Training/Test Experiments

↓

Randomly Shuffle Data

↓

Repeat LOOCV Again

↓

Repeat 10 Times
```

This greatly reduces the influence of random data ordering.

---

# What Does `print(rkf)` Show?

Printing the object displays something similar to

```text
RepeatedKFold(
    n_repeats=1,
    n_splits=21,
    random_state=None
)
```

This output simply summarizes the configuration.

It tells us:

- **21 folds** will be created.
- The complete process will be repeated **once** (or more, depending on `N_REPEATS`).
- `random_state=None` means no fixed random seed is specified for the fold generation, so repeated runs may produce different random partitions.

---

# Why Is This Important?

Notice that **no machine learning has occurred yet**.

At this stage:

```text
Dataset

↓

Create Splitting Strategy

↓

No Training Yet

↓

No Predictions Yet

↓

No Accuracy Yet
```

The object merely defines **how** future train-test splits will be generated.

Later, the notebook will iterate over these splits using

```python
for train_indices, test_indices in rkf.split(X):
```

and train a new QSVC model for each fold.

---

# Research Insight

The choice of **Repeated Leave-One-Out Cross-Validation** reflects the characteristics of the dataset.

Because only **21 alloys** are available, the authors prioritize using as much data as possible for training while still evaluating every sample independently.

This is a common strategy in materials informatics, where datasets are often too small for traditional train-test splits.

Rather than relying on one fortunate partition, the model is assessed across many carefully designed splits, producing a much more reliable estimate of its true performance.

---

# Overall Workflow

```text
Load Dataset

↓

21 Materials

↓

Create RepeatedKFold Object

↓

21 Folds (LOOCV)

↓

Repeat Entire Process Multiple Times

↓

Generate Train/Test Indices

↓

Train QSVC on Each Fold

↓

Average Performance Across All Experiments
```

---

# Key Takeaways

- **`RepeatedKFold`** → Creates a strategy for repeatedly dividing the dataset into training and testing sets.
- **No training occurs here** → This line only defines how future splits will be generated.
- **Cross-Validation** → Evaluates the model using multiple train-test partitions instead of a single split.
- **`n_splits = X.shape[0] // TEST_SIZE`** → Determines the number of folds based on the dataset size and testing set size.
- **21 folds** → Each fold contains one testing sample and twenty training samples.
- **Leave-One-Out Cross-Validation (LOOCV)** → A special case of K-Fold where every sample serves as the testing sample exactly once.
- **`n_repeats`** → Repeats the entire cross-validation process multiple times to obtain a more reliable estimate of model performance.
- **Research benefit** → LOOCV is particularly well suited for small materials-science datasets because it maximizes the amount of data available for training while still providing rigorous evaluation.


# Creating the Experiment Results Table

Before the QSVC model begins training, the notebook creates an empty **Pandas DataFrame** to store the results from every experiment.

This DataFrame does **not** perform any machine learning.

Instead, it acts as the **experiment logbook**, recording everything that happens during the training and evaluation process.

Think of it as a laboratory notebook for computational experiments.

---

# Why Create a Results Table?

During this project, the authors are not training just **one** QSVC model.

They are testing many combinations of hyperparameters, including:

- Different regularization parameters (`C`)
- Different feature map repetitions (`reps`)
- Different entanglement topologies

For every combination, the model is trained and evaluated again.

Without a structured results table, it would be extremely difficult to compare the outcomes of these experiments.

---

# Workflow

The overall workflow is:

```text
Train QSVC
      │
      ▼
Predict Test Sample
      │
      ▼
Store Results
      │
      ▼
Repeat for Every Hyperparameter Combination
      │
      ▼
Large Results Table
      │
      ▼
Statistical Analysis
      │
      ▼
Figures and Tables for the Research Paper
```

Every experiment contributes one or more new rows to this table.

---

# What Information Is Stored?

Each row of the DataFrame records the complete details of one experiment.

Typical columns include:

| Column | Purpose |
|---------|----------|
| **C** | Regularization parameter used by QSVC |
| **Reps** | Number of feature map repetitions |
| **Entanglement** | Quantum circuit connectivity (`linear`, `full`, or `circular`) |
| **Test Alloy** | Identity of the alloy used for testing |
| **True Label** | Correct class from the dataset |
| **Prediction** | Class predicted by QSVC |
| **Performance Metrics** | Evaluation results for that experiment |

A simplified example might look like:

| C | Reps | Entanglement | Test Alloy | True Label | Prediction |
|--:|------:|--------------|-------------|-----------:|-----------:|
| 0.1 | 1 | Linear | Mg-Al | 0 | 0 |
| 0.1 | 2 | Linear | Mg-Zn | 1 | 1 |
| 10 | 5 | Full | Mg-Y | 0 | 1 |

Each row represents one prediction made during one cross-validation fold.

---

# Why Store So Much Information?

It might seem sufficient to save only the final accuracy.

However, researchers usually preserve much more information because it allows deeper analysis later.

For example, they can investigate questions such as:

- Which hyperparameter combination produced the highest accuracy?
- Which alloys were consistently misclassified?
- Does increasing `reps` improve performance?
- Does the `full` entanglement topology outperform `linear`?
- Does increasing `C` improve or worsen generalization?

Without storing the individual experiment results, these questions would be impossible to answer after the experiments have finished.

---

# Traceability and Reproducibility

This DataFrame makes every experiment **traceable**.

Suppose one particular configuration produces the best performance.

Because every hyperparameter and prediction has been recorded, the authors can immediately identify exactly how that result was obtained.

For example:

```text
Best Result

↓

C = 10

↓

Reps = 3

↓

Entanglement = Full

↓

Accuracy = 95%
```

The experiment can then be reproduced simply by rerunning the notebook with those same settings.

This is an important aspect of scientific reproducibility.

---

# Statistical Analysis

After all experiments have finished, the DataFrame becomes the basis for further analysis.

Researchers can calculate:

- Mean accuracy
- Standard deviation
- Best-performing hyperparameters
- Performance comparisons
- Summary tables

They can also generate figures for publication, such as:

- Accuracy vs. `C`
- Accuracy vs. `reps`
- Accuracy for different entanglement topologies
- Confusion matrices
- Performance bar charts

Thus, the DataFrame serves as the central repository from which all later analyses are derived.

---

# Why Is This Good Research Practice?

This DataFrame is essentially the **experiment logbook** for the entire study.

Rather than storing only the final average accuracy, the authors preserve:

- The hyperparameter configuration (`C`, `reps`, and entanglement)
- The identity of the alloy being tested
- The true class label
- The predicted class
- Performance metrics

This makes the experiments transparent, reproducible, and much easier to analyse later.

---

# An Interesting Observation: R² in a QSVC Notebook

The notebook includes columns such as:

```text
R² Train

R² Test
```

This is somewhat unusual.

The **coefficient of determination (R²)** is fundamentally a **regression metric**.

It measures how well predicted **continuous numerical values** match the true values.

QSVC, however, is a **classification algorithm**.

Its outputs are discrete class labels, such as:

```text
0

or

1
```

rather than continuous numerical predictions.

Consequently, metrics commonly used for QSVC include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (for binary classification)

Using **R²** in a pure classification setting is generally not standard practice.

There are a few possible explanations:

1. **Code Reuse:** The QSVC notebook may have been adapted from an earlier regression notebook (such as a QSVR project), and the R² columns were retained even though they are no longer the most appropriate metric.
2. **Internal Comparison:** The authors may have kept the same logging structure across several notebooks to simplify comparisons between classification and regression experiments.
3. **Legacy Artifact:** The R² values may simply remain in the results table without being central to the final analysis.

When reproducing the paper, it is good practice to preserve these columns exactly as the authors implemented them. Later, if you extend or improve the methodology, you could consider replacing or supplementing them with more appropriate classification metrics.

---

# Overall Workflow

```text
Choose Hyperparameters
        │
        ▼
Train QSVC
        │
        ▼
Predict Test Sample
        │
        ▼
Store:

• Hyperparameters
• Test Alloy
• True Label
• Predicted Label
• Performance Metrics

        │
        ▼
Repeat for Every Cross-Validation Fold
        │
        ▼
Complete Results DataFrame
        │
        ▼
Statistical Analysis
        │
        ▼
Research Paper Tables and Figures
```

---

# Research Insight

One characteristic of high-quality computational research is **keeping complete records**, not just reporting the final answer.

This notebook demonstrates that principle well.

The QSVC algorithm determines **how predictions are made**, while the results DataFrame determines **how those predictions are documented**.

Well-organized experiment logs make it possible to reproduce results, diagnose unexpected behaviour, compare hyperparameters, and generate publication-quality analyses long after the computations have finished.

---

# Key Takeaways

- The DataFrame acts as the **experiment logbook** for the entire QSVC study.
- It stores the hyperparameters, tested alloy, true label, predicted label, and evaluation metrics for every experiment.
- Keeping detailed records enables reproducibility, statistical analysis, and comparison of different model configurations.
- The collected data can later be used to generate tables, plots, and figures for publication.
- The inclusion of **R²** in a QSVC notebook is unusual because QSVC is a classification algorithm; it likely reflects code reuse or a common logging framework rather than a preferred classification metric.


# Generating Descriptive Experiment Filenames

This cell is **not related to the Quantum Support Vector Classifier (QSVC) algorithm itself**.

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
QSVC/result/FMR_1_R_0.1_E_['linear', 'full', 'circular']_28_19_25_1.csv
```

At first glance, this may look complicated.

However, every part of the filename carries useful information.

---

# Breaking Down the Filename

## Folder

```text
QSVC/result/
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

So the quantum feature map was applied once.

---

## Regularization Parameter

```text
R_0.1
```

means

```text
C = 0.1
```

The QSVC model used a regularization parameter of **0.1**.

---

## Entanglement

```text
E_['linear', 'full', 'circular']
```

indicates the entanglement configuration associated with the experiment.

In many cases, the filename may contain a single topology, for example:

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

If the entire list appears in the filename, it likely reflects how the authors stored the experiment configuration or generated filenames for a batch of runs.

---

## Remaining Numbers

```text
28_19_25_1
```

These values are additional experiment identifiers.

Their exact meaning depends on the implementation.

They might represent, for example:

- Random seed
- Dataset identifier
- Threshold value
- Experiment number
- Timestamp
- Fold number

To determine their precise meaning, we would need to inspect the code that constructs the filename.

---

# Why Is This Useful?

Imagine running hundreds of experiments.

Without descriptive filenames:

```text
result.csv

result_new.csv

result_final.csv

result_final2.csv

result_latest.csv
```

After a few weeks, it becomes almost impossible to remember which file corresponds to which experiment.

---

With descriptive filenames:

```text
FMR_1_R_0.1_E_linear.csv

FMR_3_R_10_E_full.csv

FMR_5_R_100_E_circular.csv
```

the configuration of every experiment is immediately obvious.

No need to open the file.

The filename itself documents the experiment.

---

# Research Workflow

The typical workflow becomes:

```text
Choose Hyperparameters

↓

Train QSVC

↓

Evaluate Model

↓

Automatically Generate Filename

↓

Save Results

↓

Repeat Hundreds of Times
```

Each experiment produces its own uniquely named file.

---

# Why Researchers Do This

Suppose the best-performing experiment produced

```text
95% Accuracy
```

Several weeks later, you want to reproduce that result.

If the filename is

```text
result.csv
```

you have no idea which settings were used.

However, if the filename is

```text
FMR_3_R_10_E_full.csv
```

you immediately know:

```text
Feature Map Repetitions = 3

↓

Regularization Parameter = 10

↓

Entanglement = Full
```

Reproducing the experiment becomes much easier.

---

# Good Computational Research Practice

This cell illustrates an important principle of scientific computing:

> **Metadata should travel with the results.**

Here, the metadata (hyperparameters) are embedded directly into the filename.

Even if someone copies the CSV file to another computer, the filename still describes how the experiment was generated.

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
FMR_1_R_0.1_E_linear.csv

FMR_3_R_10_E_full.csv

FMR_5_R_100_E_circular.csv
```

Months later, the researcher can immediately identify the hyperparameters without opening the files.

---

# Research Insight

Notice that this cell does not change:

- the quantum feature map,
- the fidelity kernel,
- the QSVC model,
- or the machine learning algorithm.

Its sole purpose is **experiment organization**.

As research projects grow larger, organization becomes increasingly important.

Many experienced researchers spend as much time designing reproducible workflows as they do developing new algorithms.

Automatic filename generation is one small but valuable example of that philosophy.

---

# Overall Workflow

```text
Choose Hyperparameters

↓

Feature Map Repetitions

↓

Regularization Parameter

↓

Entanglement Type

↓

Construct Descriptive Filename

↓

Save Results Automatically

↓

Repeat for Every Experiment
```

---

# Key Takeaways

- This cell is **not part of the QSVC algorithm**; it is part of the experiment management workflow.
- Its purpose is to generate descriptive filenames that uniquely identify each experiment.
- The filename embeds important hyperparameters such as:
  - **Feature Map Repetitions (FMR)**
  - **Regularization Parameter (R or C)**
  - **Entanglement Topology (E)**
- Descriptive filenames improve traceability, reproducibility, and organization by allowing researchers to identify an experiment without opening the corresponding file.
- Embedding metadata into filenames is a common and highly recommended practice in computational research, especially when running large batches of experiments.


# The Heart of the QSVC Notebook

This section is the **core of the entire notebook**.

Everything before this point has been preparation:

- Importing libraries
- Defining global variables
- Loading the dataset
- Preprocessing the data
- Creating helper functions
- Setting up cross-validation
- Organizing output files

Now, the notebook finally begins performing the actual **Quantum Machine Learning experiments**.

This section contains nearly every major stage of the workflow:

- Cross-Validation
- Hyperparameter Search
- Quantum Model Construction
- Model Training
- Prediction
- Performance Evaluation
- Result Collection
- Saving Results

---

# What Is the Code Trying to Accomplish?

The central research question of this notebook is:

> **"Among all possible QSVC configurations, which one classifies Mg alloys most accurately?"**

The authors do not assume that one particular QSVC configuration is optimal.

Instead, they systematically explore many combinations of hyperparameters and compare their performance.

This process is known as **hyperparameter optimization** (or **hyperparameter search**).

---

# Why Not Train Just One Model?

Suppose we choose only one configuration:

```text
C = 1

Feature Map Repetitions = 2

Entanglement = Linear
```

We train the QSVC and obtain an accuracy of

```text
85%
```

Is this the best possible model?

There is no way to know.

Perhaps

```text
C = 10
```

would achieve 92%.

Perhaps

```text
Full Entanglement
```

would outperform the linear topology.

Perhaps increasing the feature map repetitions would improve the quantum feature space.

Rather than guessing, the notebook tests every combination.

---

# Hyperparameter Search

Earlier, the notebook defined several lists:

```python
REGU_PARA_LIST

FEATURE_MAP_REPS_LIST

ENTANGLEMENT_LIST
```

Each list contains multiple candidate values.

The notebook systematically loops through every possible combination.

For example:

```text
C = 0.1

↓

Reps = 1

↓

Entanglement = Linear
```

Train and evaluate.

Then

```text
C = 0.1

↓

Reps = 1

↓

Entanglement = Full
```

Train and evaluate again.

Then

```text
C = 0.1

↓

Reps = 1

↓

Entanglement = Circular
```

Continue.

Eventually every possible configuration is tested.

---

# Cross-Validation

Each hyperparameter combination is **not** evaluated only once.

Instead, it is evaluated using **Leave-One-Out Cross-Validation (LOOCV)**.

For every fold:

```text
20 Alloys

↓

Training

↓

1 Alloy

↓

Testing
```

The testing alloy changes every iteration until every material has been tested once.

This provides a much more reliable estimate of model performance than using a single train-test split.

---

# Building the Quantum Model

For each hyperparameter combination, the notebook constructs a new QSVC model.

The process is:

```text
Choose Hyperparameters

↓

Create ZZFeatureMap

↓

Create Fidelity Quantum Kernel

↓

Construct QSVC
```

At this stage, the model has been created but has not yet learned from any data.

---

# Training the QSVC

Next,

```python
qsvc.fit(...)
```

is called.

During training:

1. Classical features are encoded into quantum states using the **ZZFeatureMap**.
2. The **Fidelity Quantum Kernel** computes similarities between all training samples.
3. The resulting kernel matrix is supplied to the classical Support Vector Machine optimization algorithm.
4. The optimization identifies the support vectors that define the decision boundary.

After this step, the model is trained.

---

# Prediction

Once training is complete, the notebook predicts:

- The training labels
- The testing label

using

```python
qsvc.predict(...)
```

The predicted labels are then compared with the true labels.

---

# Performance Evaluation

The notebook computes performance metrics for the current experiment.

These may include:

- Accuracy
- Training accuracy
- Testing accuracy
- Other stored statistics

These values indicate how well the current QSVC configuration performed.

---

# Recording the Results

The notebook stores all relevant information in the experiment DataFrame.

Each row typically contains:

- Hyperparameters
- Test alloy
- True class
- Predicted class
- Performance metrics

Nothing is discarded.

Every experiment is preserved for later analysis.

---

# Saving the Results

After all cross-validation folds and hyperparameter combinations have been completed,

the notebook writes the DataFrame to a CSV file.

The filename contains the experimental settings, allowing the results to be easily identified and reproduced later.

---

# Putting Everything Together

The complete workflow is:

```text
Entire Dataset
        │
        ▼
Leave-One-Out Cross-Validation
        │
        ▼
Choose Regularization Parameter (C)
        │
        ▼
Choose Feature Map Repetitions
        │
        ▼
Choose Entanglement Topology
        │
        ▼
Construct ZZFeatureMap
        │
        ▼
Construct Fidelity Quantum Kernel
        │
        ▼
Build QSVC Model
        │
        ▼
Train QSVC
        │
        ▼
Predict Test Sample
        │
        ▼
Evaluate Performance
        │
        ▼
Store Results
        │
        ▼
Repeat for Every Fold
        │
        ▼
Repeat for Every Hyperparameter Combination
        │
        ▼
Save Final Results
```

---

# Why Is This the "Heart" of the Notebook?

Everything before this section prepares the experiment.

Everything after this section analyses the results.

This section is where the actual **scientific computation** takes place.

It is here that the notebook repeatedly constructs, trains, evaluates, and compares hundreds (or even thousands) of QSVC models to determine which quantum machine learning configuration performs best for classifying magnesium alloy stacking fault energies.

---

# Research Insight

Although the notebook appears to train "one model," it is actually conducting a **large-scale computational experiment**.

Each unique combination of:

- Cross-validation fold
- Regularization parameter (`C`)
- Feature map repetitions (`reps`)
- Entanglement topology

creates a **new QSVC model** that is independently trained and evaluated.

This systematic exploration is known as a **hyperparameter search**, and it is a fundamental part of modern machine learning research. Rather than relying on intuition, researchers allow the data to reveal which configuration provides the best predictive performance.

---

# Key Takeaways

- This is the central section of the notebook where all Quantum Machine Learning computations occur.
- The notebook performs a **hyperparameter search** instead of training a single QSVC model.
- Each hyperparameter combination is evaluated using **Leave-One-Out Cross-Validation (LOOCV)** for robust performance estimation.
- Every experiment follows the same pipeline: construct the quantum model, train it, predict labels, evaluate performance, and record the results.
- The repeated evaluation of many QSVC configurations allows the authors to identify the combination of parameters that best classifies the Mg alloy dataset.


# The First Loop: Cross-Validation Begins

This is the first major loop of the notebook.

For the first time, the notebook begins iterating over the dataset and preparing data for model training.

Until this point, we have only:

- Loaded the dataset
- Defined helper functions
- Created the QSVC model builder
- Defined the cross-validation strategy

Now, the actual experiment begins.

---

# The Loop

```python
for train_indices, test_indices in rkf.split(X):
```

This line tells Python:

> **"For every train-test split generated by the RepeatedKFold object, perform the following steps."**

Remember that earlier we created

```python
rkf = RepeatedKFold(...)
```

That object already knows exactly how the dataset should be divided.

The loop simply asks it for one split at a time.

---

# What Does `rkf.split(X)` Return?

It **does not return the data itself.**

Instead, it returns two arrays of integers.

Example:

```text
train_indices

[0,2,3,4,5,6,7,...,20]
```

```text
test_indices

[1]
```

These numbers are simply **row numbers** in the dataset.

They tell Python:

```text
Use these rows for training.

Use this row for testing.
```

Nothing has been copied yet.

Only the locations of the samples are provided.

---

# Example

Suppose our dataset contains only five alloys.

```text
Index

0

1

2

3

4
```

During the first iteration,

`RepeatedKFold` might return

```text
train_indices

[1,2,3,4]
```

```text
test_indices

[0]
```

Meaning:

```text
Training

Sample 1

Sample 2

Sample 3

Sample 4
```

```text
Testing

Sample 0
```

On the next iteration,

```text
train_indices

[0,2,3,4]
```

```text
test_indices

[1]
```

Now Sample 1 becomes the testing sample.

This continues until every sample has been used for testing once.

---

# Preparing the Dataset

Immediately after obtaining the indices, the notebook calls

```python
X_train, y_train, X_test, y_test, element_test, element_train = prepare_dataset_k_fold(
    X,
    y,
    train_indices,
    test_indices
)
```

This function performs **all preprocessing** for the current fold.

Notice that the loop itself stays very clean.

Instead of writing dozens of preprocessing commands repeatedly,

everything is delegated to a helper function.

---

# What Happens Inside `prepare_dataset_k_fold()`?

Earlier, we studied this function in detail.

It performs several important operations:

### 1. Split the Dataset

Using the training and testing indices,

the dataset is divided into

```text
Training Set

↓

Testing Set
```

---

### 2. Separate the Element Names

The first column contains

```text
Mg-Al

Mg-Zn

Mg-Y
```

These are useful for identifying alloys,

but they cannot be used as numerical inputs for QSVC.

The function therefore extracts them separately.

Result:

```text
element_train
```

and

```text
element_test
```

---

### 3. Remove the Element Column

The remaining numerical features are

```text
Electronegativity

Bulk Modulus

Volume
```

These become

```python
X_train

X_test
```

which are suitable for machine learning.

---

### 4. Scale the Features

The function applies

```python
MinMaxScaler
```

to transform all features into the range

```text
-1

↓

1
```

This ensures that all input variables have comparable numerical scales before being encoded into quantum states.

---

### 5. Return Everything Needed

Finally, the function returns six objects:

```text
Training Features

↓

X_train
```

```text
Training Labels

↓

y_train
```

```text
Testing Features

↓

X_test
```

```text
Testing Labels

↓

y_test
```

```text
Training Element Names

↓

element_train
```

```text
Testing Element Names

↓

element_test
```

Everything required for the current experiment is now available.

---

# Why Use a Separate Function?

Imagine placing all preprocessing code directly inside the loop.

The notebook would become long and difficult to read.

Instead,

the authors package the preprocessing into one reusable function.

This has several advantages:

- Cleaner code
- Easier debugging
- Less repetition
- Better organization
- Improved readability

This is a common software engineering practice in research code.

---

# What Happens Next?

Once this function finishes,

the notebook has everything needed for one experiment.

The next steps will be:

```text
Prepared Training Data

↓

Build QSVC

↓

Train QSVC

↓

Predict Labels

↓

Store Results
```

Then the entire process repeats for the next cross-validation fold.

---

# Overall Workflow

```text
RepeatedKFold

↓

Generate Training Indices

↓

Generate Testing Indices

↓

prepare_dataset_k_fold()

↓

Split Dataset

↓

Separate Element Names

↓

Remove Text Columns

↓

Scale Numerical Features

↓

Return

• X_train

• y_train

• X_test

• y_test

• element_train

• element_test

↓

Ready for QSVC Training
```

---

# Research Insight

This loop represents the transition from **experimental design** to **experimental execution**.

The `RepeatedKFold` object determines **which samples** belong to the training and testing sets, while `prepare_dataset_k_fold()` transforms those raw samples into the numerical format required by the QSVC model.

Separating these responsibilities—one object handling data partitioning and another handling preprocessing—is an example of good research software design. It keeps the code modular, easier to verify, and simpler to extend if the preprocessing strategy changes in future studies.

---

# Key Takeaways

- `rkf.split(X)` generates the **training and testing indices** for each cross-validation fold.
- The indices identify **which rows** belong to the training and testing sets; they do not contain the data itself.
- `prepare_dataset_k_fold()` performs all preprocessing for the current fold.
- The preprocessing includes:
  - Splitting the dataset
  - Extracting element names
  - Removing the non-numerical element column
  - Scaling the numerical features
- The function returns six objects: `X_train`, `y_train`, `X_test`, `y_test`, `element_train`, and `element_test`.
- After this step, the data are fully prepared for constructing and training the QSVC model.


# The Second Loop: Exploring the Regularization Parameter (`C`)

After preparing the training and testing datasets for one cross-validation fold, the notebook enters its second loop.

```python
for C_value in REGU_PARA_LIST:
```

This loop is responsible for testing different values of the **regularization parameter (`C`)** used by the Quantum Support Vector Classifier (QSVC).

Instead of assuming that one value of `C` is optimal, the notebook systematically evaluates several possibilities.

---

# Where Does `REGU_PARA_LIST` Come From?

Earlier in the notebook, the following global variable was defined:

```python
REGU_PARA_LIST = [0.1, 1, 10, 100]
```

This list contains the candidate values of the regularization parameter that will be tested.

The loop simply goes through each value one at a time.

---

# What Happens During the Loop?

The loop executes four times.

The iterations are:

```text
Iteration 1

↓

C = 0.1
```

```text
Iteration 2

↓

C = 1
```

```text
Iteration 3

↓

C = 10
```

```text
Iteration 4

↓

C = 100
```

Each iteration represents a **new QSVC experiment**.

---

# What Is the Regularization Parameter?

The parameter **`C`** controls the balance between:

- fitting the training data closely, and
- maintaining a simple, generalizable decision boundary.

It is inherited from the classical Support Vector Machine (SVM), and QSVC uses the same concept.

---

# Small `C`

For example,

```text
C = 0.1
```

A small value means the classifier is **more tolerant of mistakes** on the training data.

Instead of trying to classify every training sample perfectly, it prefers a simpler decision boundary.

Conceptually:

```text
Simpler Decision Boundary

↓

Less Sensitive to Noise

↓

Better Generalization

↓

Possible Underfitting
```

---

# Large `C`

Now consider

```text
C = 100
```

A large value penalizes training errors much more strongly.

The classifier attempts to classify nearly every training sample correctly.

Conceptually:

```text
More Complex Decision Boundary

↓

Fits Training Data More Closely

↓

Higher Risk of Overfitting
```

---

# Why Test Multiple Values?

Before running the experiment, the authors do not know which value of `C` will perform best for the magnesium alloy dataset.

Instead of relying on intuition, they allow the data to determine the best choice.

For every cross-validation fold, the notebook evaluates:

```text
C = 0.1
```

↓

Train QSVC

↓

Evaluate Performance

↓

Store Results

Then

```text
C = 1
```

↓

Train Again

↓

Evaluate Again

↓

Store Results

This process continues until every value in `REGU_PARA_LIST` has been tested.

---

# Relationship with Cross-Validation

Notice the nesting of the loops.

The first loop selects the training and testing samples.

Inside that loop, the second loop tests every value of `C`.

The structure looks like:

```text
Cross-Validation Fold 1

    ↓

    C = 0.1

    C = 1

    C = 10

    C = 100

↓

Cross-Validation Fold 2

    ↓

    C = 0.1

    C = 1

    C = 10

    C = 100

↓

Continue Until Every Fold Is Complete
```

This means that every fold is evaluated using **all candidate regularization parameters**.

---

# Why Is This Important?

Changing `C` changes the behaviour of the QSVC.

Even though the:

- dataset,
- feature map,
- quantum kernel,
- and cross-validation split

remain the same,

the learned decision boundary may be completely different.

Testing multiple values allows the researchers to identify the regularization strength that provides the best balance between fitting the training data and generalizing to unseen alloys.

---

# Research Workflow

```text
Receive Training/Test Split

↓

Choose C = 0.1

↓

Build QSVC

↓

Train

↓

Evaluate

↓

Store Results

↓

Choose C = 1

↓

Repeat

↓

Choose C = 10

↓

Repeat

↓

Choose C = 100

↓

Repeat
```

Only after all four values have been evaluated does the notebook move on to the next cross-validation fold.

---

# Research Insight

The regularization parameter is one of the most influential hyperparameters in Support Vector Machines.

Rather than assuming a "correct" value, the authors perform a **hyperparameter search**, allowing each candidate value to compete under identical cross-validation conditions.

This approach makes the comparison fair and scientifically rigorous, ensuring that the final choice of `C` is supported by experimental evidence rather than personal preference.

---

# Overall Workflow

```text
Current Cross-Validation Fold

        │
        ▼
Choose Regularization Parameter (C)

        │
        ├──────────────► C = 0.1
        │
        ├──────────────► C = 1
        │
        ├──────────────► C = 10
        │
        └──────────────► C = 100

        │
        ▼
Build QSVC

        ▼
Train

        ▼
Evaluate

        ▼
Store Results

        ▼
Repeat for Next C Value
```

---

# Key Takeaways

- The second loop iterates over the **regularization parameter (`C`)**.
- `REGU_PARA_LIST = [0.1, 1, 10, 100]` means the notebook evaluates **four different QSVC models** for each cross-validation fold.
- A **small `C`** encourages a simpler decision boundary and is more tolerant of training errors.
- A **large `C`** fits the training data more closely but may increase the risk of overfitting.
- Testing multiple values of `C` is an example of **hyperparameter optimization**, allowing the researchers to determine which setting produces the best classification performance.
- Every value of `C` is evaluated under the same cross-validation conditions to ensure a fair comparison.


# The Third Loop: Exploring Feature Map Repetitions (`reps`)

After selecting one cross-validation fold and one regularization parameter (`C`), the notebook enters the third loop.

```python
for feature_map_reps in FEATURE_MAP_REPS_LIST:
```

This loop investigates one of the **quantum-specific hyperparameters** of the QSVC model:

> **How many times should the quantum feature map be repeated?**

Unlike the regularization parameter (`C`), which comes from the classical Support Vector Machine, **`reps` is unique to Quantum Machine Learning** because it controls the structure of the quantum circuit used to encode the classical data.

---

# Where Does `FEATURE_MAP_REPS_LIST` Come From?

Earlier in the notebook, the following variable was defined:

```python
FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]
```

This means the notebook will test five different quantum circuits.

Each value represents a different number of repetitions of the `ZZFeatureMap`.

---

# What Happens During the Loop?

The loop executes five times.

```text
Iteration 1

↓

Repetitions = 1
```

```text
Iteration 2

↓

Repetitions = 2
```

```text
Iteration 3

↓

Repetitions = 3
```

```text
Iteration 4

↓

Repetitions = 4
```

```text
Iteration 5

↓

Repetitions = 5
```

Each repetition count produces a **new QSVC model** with a different quantum circuit.

---

# What Does `reps` Mean?

The QSVC does not work directly with the raw numerical features.

Instead, the classical data are first encoded into a quantum state using the **ZZFeatureMap**.

A single repetition (`reps = 1`) means the encoding circuit is applied once.

```text
Classical Features

↓

ZZFeatureMap

↓

Quantum State
```

If

```text
reps = 2
```

the same encoding pattern is applied twice.

```text
Classical Features

↓

ZZFeatureMap

↓

ZZFeatureMap

↓

Quantum State
```

For

```text
reps = 5
```

the feature map is repeated five times.

This creates a deeper quantum circuit.

---

# Why Repeat the Feature Map?

Repeating the feature map increases the complexity of the quantum encoding.

In general:

```text
Higher reps

↓

Deeper Quantum Circuit

↓

More Feature Interactions

↓

Potentially Richer Quantum Feature Space
```

A richer feature space may help the QSVC distinguish between materials that are difficult to separate.

However, increasing the circuit depth also has potential drawbacks:

- Longer execution time
- Higher computational cost
- Greater circuit complexity
- On real quantum hardware, increased sensitivity to noise

Even when using a simulator, deeper circuits require more computational resources.

---

# Why Test Multiple Values?

Before performing the experiments, the authors do not know which circuit depth will provide the best classification performance.

Instead of selecting a single value, they evaluate all candidates.

For each value of `C`, the notebook tests:

```text
Reps = 1

↓

Build QSVC

↓

Train

↓

Evaluate
```

Then

```text
Reps = 2
```

↓

Build Another QSVC

↓

Train Again

↓

Evaluate Again

This continues until all five repetition values have been explored.

---

# Relationship with Previous Loops

Remember that this loop is nested inside the previous loops.

The execution order is now:

```text
Cross-Validation Fold

↓

Choose C

↓

Choose Feature Map Repetitions
```

For example:

```text
Fold 1

↓

C = 0.1

↓

Reps = 1

↓

Train
```

Then

```text
Fold 1

↓

C = 0.1

↓

Reps = 2

↓

Train
```

Eventually,

```text
Fold 1

↓

C = 1

↓

Reps = 1

↓

Train
```

The process continues until every combination of fold, `C`, and `reps` has been evaluated.

---

# Why Is This Important?

Unlike the regularization parameter, which influences the classical optimization process, `reps` changes the **quantum circuit itself**.

Changing the number of repetitions changes:

- the depth of the quantum circuit,
- how classical information is encoded,
- and ultimately the quantum kernel computed by the Fidelity Quantum Kernel.

Therefore, two QSVC models with identical datasets and identical `C` values can behave differently simply because they use different feature map depths.

---

# Research Workflow

```text
Current Cross-Validation Fold

↓

Current Regularization Parameter (C)

↓

Choose Repetitions = 1

↓

Build QSVC

↓

Train

↓

Evaluate

↓

Store Results

↓

Choose Repetitions = 2

↓

Repeat

↓

...

↓

Choose Repetitions = 5

↓

Repeat
```

---

# Research Insight

The number of feature map repetitions is one of the most important **quantum hyperparameters** in kernel-based quantum machine learning.

Increasing `reps` generally increases the expressive power of the quantum feature map, allowing it to represent more complex relationships between input features.

However, greater expressivity does not automatically lead to better performance. Deeper circuits may also become more computationally expensive and, on real quantum hardware, more susceptible to noise.

By evaluating several values of `reps`, the authors allow the experimental results to determine the most effective circuit depth for the Mg alloy classification task.

---

# Overall Workflow

```text
Current Cross-Validation Fold

        │
        ▼
Current Regularization Parameter (C)

        │
        ▼
Choose Feature Map Repetitions

        │
        ├──────────────► reps = 1
        │
        ├──────────────► reps = 2
        │
        ├──────────────► reps = 3
        │
        ├──────────────► reps = 4
        │
        └──────────────► reps = 5

        │
        ▼
Build QSVC

        ▼
Train

        ▼
Evaluate

        ▼
Store Results

        ▼
Repeat for Next Repetition Value
```

---

# Key Takeaways

- The third loop iterates over the **feature map repetition parameter (`reps`)**.
- `FEATURE_MAP_REPS_LIST = [1, 2, 3, 4, 5]` means the notebook evaluates **five different quantum circuit depths**.
- Increasing `reps` repeats the `ZZFeatureMap`, creating a deeper and potentially more expressive quantum circuit.
- Different repetition values produce different quantum kernels and may lead to different classification performance.
- Testing multiple values of `reps` is part of the notebook's **quantum hyperparameter search**, allowing the researchers to identify the circuit depth that works best for the magnesium alloy dataset.



# The Fourth Loop: Exploring Entanglement Topologies

After selecting:

- a cross-validation fold,
- a regularization parameter (`C`),
- and a feature map repetition (`reps`),

the notebook enters the fourth and final hyperparameter loop.

```python
for entanglement in ENTANGLEMENT_LIST:
```

This loop explores another **quantum-specific hyperparameter**:

> **How should the qubits be connected during the feature encoding process?**

The answer determines the **entanglement topology** used by the `ZZFeatureMap`.

---

# Where Does `ENTANGLEMENT_LIST` Come From?

Earlier in the notebook, the following variable was defined:

```python
ENTANGLEMENT_LIST = ['linear', 'full', 'circular']
```

This means the notebook will evaluate three different quantum circuit structures.

The loop executes once for each topology.

---

# The Three Experiments

The loop performs three iterations.

```text
Iteration 1

↓

Entanglement = Linear
```

```text
Iteration 2

↓

Entanglement = Full
```

```text
Iteration 3

↓

Entanglement = Circular
```

Each iteration creates a completely new quantum circuit.

---

# What Is Entanglement?

Earlier we learned that entanglement determines **which qubits interact with one another**.

Changing the entanglement pattern changes the structure of the quantum feature map.

For three qubits:

### Linear

```text
Q0 ─── Q1 ─── Q2
```

Each qubit interacts only with its nearest neighbour.

This produces the simplest circuit.

---

### Full

```text
Q0 ───── Q1
│ \     /
│  \   /
│   \ /
Q2 ────
```

Every qubit interacts with every other qubit.

This produces the richest feature interactions but also the most complex circuit.

---

### Circular

```text
Q0 ─── Q1
│       │
└── Q2 ─┘
```

The first and last qubits are also connected, forming a ring.

This provides an intermediate level of connectivity.

---

# Why Test Different Topologies?

The authors do not know beforehand which interaction pattern best captures the relationships between the alloy features.

Instead of choosing one topology,

they evaluate all three.

For every combination of:

- Cross-validation fold
- Regularization parameter (`C`)
- Feature map repetitions (`reps`)

the notebook additionally tests

```text
Linear
```

↓

Train

↓

Evaluate

↓

Store Results

Then

```text
Full
```

↓

Train Again

↓

Evaluate Again

↓

Store Results

Then

```text
Circular
```

↓

Train Again

↓

Evaluate Again

↓

Store Results

---

# How Many QSVC Models Are Trained?

At this point, all four nested loops combine.

For one cross-validation fold:

```text
Regularization Parameters

4 choices

×

Feature Map Repetitions

5 choices

×

Entanglement Topologies

3 choices
```

Total:

```text
4 × 5 × 3 = 60
```

So,

**60 different QSVC models** are trained for **every single train-test split**.

---

Earlier, we established that the dataset contains

```text
21 samples
```

Using Leave-One-Out Cross-Validation,

there are

```text
21 folds
```

Therefore,

```text
21 × 60 = 1260
```

QSVC models are trained during **one complete cross-validation cycle**.

Finally,

```python
N_REPEATS = 10
```

means the entire LOOCV procedure is repeated ten times.

Therefore,

```text
1260 × 10 = 12,600
```

**QSVC training runs** are performed in total.

This highlights why automation is essential—manually performing thousands of experiments would be impossible.

---

# Monitoring Progress

During each iteration, the notebook prints

```python
print(
    f"C:{C_value} "
    f"feature_map_reps:{feature_map_reps} "
    f"entanglement:{entanglement}"
)
```

This serves no machine learning purpose.

Instead, it allows the researcher to monitor which experiment is currently running.

Example:

```text
C:10 feature_map_reps:3 entanglement:full
```

When running thousands of experiments, this kind of progress output is extremely useful for debugging and tracking long computations.

---

# Building a New QSVC

Next, the notebook calls

```python
qsvc = reconfig_quantum_kernel_qsvc(
    feature_dimension=NUM_FEATURES,
    C=C_value,
    reps=feature_map_reps,
    entangle=entanglement
)
```

We studied this function earlier.

Internally it performs

```text
Classical Features

↓

ZZFeatureMap

↓

Fidelity Quantum Kernel

↓

QSVC
```

Every iteration constructs a **brand-new quantum classifier**.

Nothing from the previous iteration is reused.

This ensures that each hyperparameter combination is evaluated independently.

---

# Training the Model

The notebook then calls

```python
predict_train, predict_test = train_qsvc(
    qsvc,
    X_train,
    y_train,
    X_test
)
```

Inside this function:

```text
Train QSVC

↓

Learn from Training Data

↓

Predict Training Labels

↓

Predict Testing Labels
```

The training predictions indicate how well the model fits the training data,

while the testing predictions measure how well the model generalizes to unseen alloys.

---

# Converting Predictions Back

Earlier, the target labels were transformed using

```python
MinMaxScaler
```

which mapped

```text
0

↓

-1
```

and

```text
1

↓

1
```

After prediction, the notebook converts the labels back:

```python
all_preds = y_scaler.inverse_transform(...)
all_targets = y_scaler.inverse_transform(...)
```

This restores the original class representation:

```text
-1

↓

0
```

```text
1

↓

1
```

These conversions make the stored results easier to interpret.

---

# Recording the Experiment

The notebook creates a dictionary called

```python
new_row
```

This dictionary contains everything produced during the experiment.

Typical entries include:

```text
Regularization Parameter

↓

C
```

```text
Feature Map Repetitions

↓

reps
```

```text
Entanglement Topology

↓

entanglement
```

```text
Testing Alloy

↓

element test
```

```text
True Testing Label

↓

actual test
```

```text
Predicted Testing Label

↓

predicted test
```

```text
Training Alloy

↓

element train
```

```text
Training Prediction

↓

predicted train
```

along with performance metrics.

Nothing is discarded.

Every experiment is fully documented.

---

# Why Is Only R² Train Stored?

The notebook includes

```python
R2 train
```

but

```python
R2 test
```

is commented out.

This is another indication that the notebook likely evolved from a **regression project**.

For classification problems,

metrics such as

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

are much more common than the coefficient of determination (R²).

---

# Adding the Row

Once the dictionary is complete,

the notebook executes

```python
df.loc[len(df)] = new_row
```

This simply appends the experiment to the results DataFrame.

Every QSVC training run contributes one additional row.

After thousands of experiments,

the DataFrame becomes a complete record of the study.

---

# Saving the Results

Finally,

```python
df.to_csv(file_name, index=False)
```

writes the DataFrame to disk.

Notice that the CSV is updated **after every experiment**.

This is good research practice.

If the notebook crashes after several hours,

the completed experiments have already been saved and are not lost.

---

# Additional Metadata

The statement

```python
df.at[0, "info"]
```

adds extra information to the first row of the DataFrame.

Researchers often use this field to store metadata such as:

- dataset information,
- experiment description,
- software version,
- or notes about the run.

This makes the CSV file more self-contained.

---

# Complete Workflow

```text
Cross-Validation Fold
        │
        ▼
Choose C
        │
        ▼
Choose Feature Map Repetitions
        │
        ▼
Choose Entanglement Topology
        │
        ▼
Build ZZFeatureMap
        │
        ▼
Create Fidelity Quantum Kernel
        │
        ▼
Construct QSVC
        │
        ▼
Train QSVC
        │
        ▼
Predict Training and Testing Labels
        │
        ▼
Convert Labels Back
        │
        ▼
Create Results Dictionary
        │
        ▼
Append Row to DataFrame
        │
        ▼
Update CSV File
        │
        ▼
Repeat Until All 12,600 Experiments Finish
```

---

# Research Insight

This nested loop is the computational engine of the entire paper.

Rather than evaluating a single QSVC model, the notebook performs a **systematic hyperparameter search** across every combination of:

- Cross-validation fold,
- Regularization parameter (`C`),
- Feature map repetitions (`reps`),
- and entanglement topology.

Each combination results in a completely new quantum classifier that is independently trained and evaluated. The outcome is a comprehensive experimental study that allows the authors to compare quantum circuit designs objectively and identify the configuration that performs best on the Mg alloy classification problem.

---

# Key Takeaways

- The fourth loop iterates over the three entanglement topologies: **`linear`**, **`full`**, and **`circular`**.
- Combining all hyperparameter loops gives **4 × 5 × 3 = 60 QSVC models** for every cross-validation fold.
- With **21 Leave-One-Out folds**, the notebook trains **1,260 QSVC models** in one complete evaluation cycle.
- Repeating the entire procedure **10 times** results in **12,600 independent QSVC training runs**.
- Each iteration:
  - builds a new QSVC,
  - trains it,
  - predicts training and testing labels,
  - records all results,
  - and immediately updates the CSV file.
- Saving the results after every experiment improves robustness by preserving completed work even if the program is interrupted.


# The Complete QSVC Experimental Pipeline

This diagram summarizes the **entire workflow** of the notebook.

Rather than viewing each code cell independently, it helps to see how every stage connects to the next.

The notebook follows a structured machine learning pipeline, beginning with raw data and ending with a fully documented set of experimental results.

Every experiment follows exactly the same sequence of steps.

---

# 1. Load the Dataset

The experiment begins by reading the dataset from a CSV file.

```text
CSV File

↓

Pandas DataFrame

↓

Input Features (X)

↓

Target Labels (y)
```

The selected input features are:

- Electronegativity
- Bulk Modulus
- Atomic Volume

The target variable is:

- Stacking Fault Energy (SFE)

Since QSVC is a classifier, the continuous SFE values are converted into binary class labels using the chosen threshold.

---

# 2. Cross-Validation

Next, the notebook creates a **Repeated Leave-One-Out Cross-Validation** strategy.

```text
Entire Dataset

↓

Repeated K-Fold

↓

One Training/Test Split
```

Each alloy becomes the testing sample once during every repetition.

This ensures that every material contributes equally to the evaluation.

---

# 3. Train-Test Split

For the current fold,

the dataset is divided into

```text
Training Data

↓

Testing Data
```

At the same time,

the element names are separated from the numerical features because the QSVC cannot process text.

---

# 4. Feature Scaling

The numerical features are normalized using

```python
MinMaxScaler
```

which transforms every feature into the interval

```text
[-1, 1]
```

Scaling ensures that all features contribute fairly to the quantum encoding process.

---

# 5. Hyperparameter Search

The notebook now begins exploring every QSVC configuration.

Three nested loops perform a systematic hyperparameter search.

---

## Regularization Parameter (`C`)

```text
0.1

↓

1

↓

10

↓

100
```

The notebook evaluates four different levels of regularization.

---

## Feature Map Repetitions (`reps`)

```text
1

↓

2

↓

3

↓

4

↓

5
```

Each value produces a different quantum circuit depth.

---

## Entanglement Topology

```text
Linear

↓

Full

↓

Circular
```

Each topology defines a different pattern of qubit interactions.

Together, these three loops generate

```text
4 × 5 × 3 = 60
```

different QSVC models for every cross-validation fold.

---

# 6. Build the Quantum Kernel

Using the selected hyperparameters,

the notebook constructs

```text
ZZFeatureMap

↓

Fidelity Quantum Kernel
```

The feature map encodes classical data into quantum states,

while the fidelity kernel measures the similarity between those states.

---

# 7. Create the QSVC Model

The quantum kernel is then supplied to

```python
QSVC(...)
```

creating a new Quantum Support Vector Classifier.

At this point,

the model has been configured but has **not yet learned from the data**.

---

# 8. Train the Model

The notebook calls

```python
qsvc.fit(...)
```

During training,

the following operations occur internally:

```text
Encode Training Samples

↓

Compute Quantum Kernel Matrix

↓

Solve Classical SVM Optimization

↓

Learn Decision Boundary
```

The QSVC is now trained.

---

# 9. Predict

Once training is complete,

the notebook predicts

- Training labels
- Testing labels

using

```python
qsvc.predict(...)
```

The testing predictions are especially important because they measure how well the model generalizes to unseen alloys.

---

# 10. Inverse Scaling

Earlier,

the target labels were transformed using a scaler.

Before storing the predictions,

the notebook converts them back to their original representation.

```text
Scaled Labels

↓

Inverse Transform

↓

Original Labels
```

This makes the saved results easier to interpret.

---

# 11. Store the Results

The notebook creates a dictionary containing information such as:

- Regularization parameter
- Feature map repetitions
- Entanglement topology
- Testing alloy
- Training alloy
- True labels
- Predicted labels
- Performance metrics

This dictionary becomes one row in the experiment DataFrame.

---

# 12. Save the Results

Finally,

the DataFrame is written to a CSV file.

```text
Experiment Results

↓

DataFrame

↓

CSV File
```

The CSV file is updated after every experiment,

ensuring that completed work is preserved even if the notebook stops unexpectedly.

---

# 13. Repeat

Once one experiment is finished,

the notebook immediately begins the next one.

The process repeats until every combination of:

- Cross-validation fold
- Regularization parameter
- Feature map repetitions
- Entanglement topology

has been evaluated.

---

# Complete Experimental Workflow

```text
Load Dataset
      │
      ▼
Create Input Features (X)
and Target Labels (y)
      │
      ▼
Repeated Leave-One-Out
Cross-Validation
      │
      ▼
Train/Test Split
      │
      ▼
Scale Features
      │
      ▼
Choose Regularization Parameter (C)
      │
      ▼
Choose Feature Map Repetitions
      │
      ▼
Choose Entanglement Topology
      │
      ▼
Construct ZZFeatureMap
      │
      ▼
Build Fidelity Quantum Kernel
      │
      ▼
Create QSVC
      │
      ▼
Train Model
      │
      ▼
Predict Labels
      │
      ▼
Inverse Transform Predictions
      │
      ▼
Record Results
      │
      ▼
Append to DataFrame
      │
      ▼
Save CSV
      │
      ▼
Next Experiment
```

---

# How Many Experiments Are Performed?

For this notebook:

```text
Regularization Parameters

4

×

Feature Map Repetitions

5

×

Entanglement Topologies

3

=

60 Models per Fold
```

There are

```text
21
```

cross-validation folds.

Therefore,

```text
60 × 21 = 1,260
```

QSVC models are trained during one complete evaluation cycle.

Since

```python
N_REPEATS = 10
```

the entire procedure is repeated ten times.

Total:

```text
1,260 × 10 = 12,600
```

independent QSVC training runs.

---

# Research Insight

Although the notebook appears to be a single machine learning script, it is actually an **automated experimental framework**.

Each pass through the nested loops represents a new scientific experiment with its own quantum circuit, hyperparameters, predictions, and recorded outcomes. By systematically evaluating thousands of QSVC configurations under identical cross-validation conditions, the authors ensure that their conclusions are based on comprehensive experimental evidence rather than a single trial.

This workflow reflects good computational research practice: **automate the experiments, record every result, and let the data determine the best-performing model.**

---

# Key Takeaways

- The notebook follows a complete end-to-end Quantum Machine Learning pipeline, from loading the dataset to saving the final results.
- Cross-validation ensures that every alloy is evaluated as a testing sample.
- Three nested hyperparameter loops explore different values of:
  - Regularization parameter (`C`)
  - Feature map repetitions (`reps`)
  - Entanglement topology
- Each combination produces a new QSVC model that is independently trained and evaluated.
- Results are recorded after every experiment and immediately saved to a CSV file, ensuring reproducibility and protecting against data loss.
- In total, the notebook performs **12,600 independent QSVC training runs**, making it a large-scale automated computational experiment rather than a single model training task.



# Final Conclusive Workflow of the QSVC Notebook

After understanding every section of the notebook individually, we can now view the entire workflow as one complete computational experiment.

The notebook is designed to answer a single scientific question:

> **Which Quantum Support Vector Classifier (QSVC) configuration best classifies magnesium alloys into low and high stacking fault energy (SFE) categories?**

Rather than training a single model, the notebook systematically evaluates thousands of QSVC configurations using rigorous cross-validation and records every result for later analysis.

---

# Step 1: Load the Alloy Dataset

The experiment begins by reading the alloy dataset from a CSV file.

The dataset contains:

- **Element** (alloy name)
- **Electronegativity**
- **Bulk Modulus**
- **Atomic Volume**
- **Stacking Fault Energy (SFE)**

The numerical properties become the **input features**, while the SFE values serve as the prediction target.

```text
CSV Dataset
      │
      ▼
Input Features (X)
      │
      ▼
Target Variable (y)
```

---

# Step 2: Convert the Regression Target into Binary Classes

Originally, the stacking fault energy is a **continuous numerical value**, making the problem a regression task.

Example:

```text
15.8

22.4

18.9

27.3
```

However, QSVC is a **classification algorithm**, so the notebook converts the continuous SFE values into two classes using a predefined threshold.

For example:

```text
SFE < 19

↓

Class 0 (Low SFE)
```

```text
SFE ≥ 19

↓

Class 1 (High SFE)
```

This transforms the original regression problem into a **binary classification problem**.

---

# Step 3: Evaluate Using Leave-One-Out Cross-Validation

Instead of performing a single train-test split, the notebook uses **Leave-One-Out Cross-Validation (LOOCV)**.

For each iteration:

- One alloy becomes the testing sample.
- The remaining alloys become the training set.

```text
21 Alloys

↓

20 Training

+

1 Testing
```

This process repeats until every alloy has been used as the testing sample.

To further improve reliability, the entire LOOCV procedure is repeated multiple times using `RepeatedKFold`.

This provides a robust estimate of model performance on a small dataset.

---

# Step 4: Explore Every QSVC Configuration

The notebook performs a systematic **hyperparameter search**.

Three nested loops explore every combination of:

### Regularization Parameter (`C`)

```text
0.1

1

10

100
```

### Feature Map Repetitions (`reps`)

```text
1

2

3

4

5
```

### Entanglement Topology

```text
Linear

Full

Circular
```

Together, these produce:

```text
4 × 5 × 3 = 60
```

different QSVC models for every cross-validation fold.

Each combination creates a **new quantum classifier** with its own quantum circuit.

---

# Step 5: Build and Train the QSVC

For each hyperparameter combination, the notebook:

1. Constructs a `ZZFeatureMap`.
2. Builds a `FidelityQuantumKernel`.
3. Creates a new `QSVC`.
4. Trains the classifier using the current training set.

Internally, the workflow is:

```text
Classical Features
        │
        ▼
ZZFeatureMap
        │
        ▼
Quantum States
        │
        ▼
Fidelity Quantum Kernel
        │
        ▼
Classical SVM Optimization
        │
        ▼
Trained QSVC
```

Every hyperparameter combination is trained independently.

---

# Step 6: Predict Both Training and Testing Alloys

Once the model has learned from the training data, it predicts:

- the labels of the training alloys, and
- the label of the unseen testing alloy.

These predictions allow the notebook to evaluate:

- how well the model fits the training data, and
- how well it generalizes to new materials.

The predicted labels are then converted back to their original representation before being stored.

---

# Step 7: Record Every Experiment

Each completed experiment is stored as a new row in the results DataFrame.

The recorded information includes:

- Regularization parameter (`C`)
- Feature map repetitions (`reps`)
- Entanglement topology
- Testing alloy
- Training alloy
- True labels
- Predicted labels
- Performance metrics

Nothing is discarded.

Every experiment contributes to the final dataset used for analysis.

---

# Step 8: Continuously Save the Results

After each experiment, the DataFrame is immediately written to a CSV file.

```text
Experiment

↓

Update DataFrame

↓

Save CSV
```

Saving after every iteration ensures that results are preserved even if the program is interrupted during a long computation.

This is an important aspect of robust computational research.

---

# Complete Workflow

```text
Load Alloy Dataset
        │
        ▼
Create Input Features (X)
and Target Labels (y)
        │
        ▼
Convert Continuous SFE
into Binary Classes
        │
        ▼
Leave-One-Out Cross-Validation
        │
        ▼
Prepare Training and Testing Data
        │
        ▼
Scale Numerical Features
        │
        ▼
Choose Regularization Parameter (C)
        │
        ▼
Choose Feature Map Repetitions (reps)
        │
        ▼
Choose Entanglement Topology
        │
        ▼
Build ZZFeatureMap
        │
        ▼
Construct Fidelity Quantum Kernel
        │
        ▼
Create QSVC
        │
        ▼
Train QSVC
        │
        ▼
Predict Training and Testing Labels
        │
        ▼
Convert Predictions Back
        │
        ▼
Record Results
        │
        ▼
Append to DataFrame
        │
        ▼
Update CSV File
        │
        ▼
Repeat Until Every Configuration
Has Been Evaluated
```

---

# Final Research Insight

This notebook is much more than an implementation of the QSVC algorithm.

It is a **fully automated experimental framework** for evaluating quantum machine learning models.

By combining rigorous cross-validation with a systematic exploration of classical and quantum hyperparameters, the notebook performs thousands of controlled experiments under identical conditions. Every result is recorded, allowing the researchers to compare different quantum circuit designs objectively and identify the configuration that best classifies magnesium alloy stacking fault energies.

This workflow demonstrates an essential principle of computational research:

> **Do not rely on a single experiment. Design an automated pipeline that explores the problem systematically, records every outcome, and lets the experimental evidence determine the best-performing model.**

---

# Key Takeaways

- The notebook converts a **regression dataset** into a **binary classification problem** suitable for QSVC.
- **Leave-One-Out Cross-Validation (LOOCV)** provides a rigorous evaluation strategy for the small alloy dataset.
- Three nested hyperparameter loops explore every combination of:
  - Regularization parameter (`C`)
  - Feature map repetitions (`reps`)
  - Entanglement topology
- Each hyperparameter combination produces a new QSVC model that is independently built, trained, and evaluated.
- Every prediction and performance metric is recorded in a DataFrame and continuously saved to a CSV file.
- The notebook functions as a **large-scale automated computational experiment**, enabling reproducible and objective comparison of quantum machine learning models for Mg alloy classification.