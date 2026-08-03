# Week 1: [Dates, June 13 - June 19]

##  Weekly Goals
- [ ] Complete Qiskit basic tutorials (gates, Bloch sphere visualization)
- [ ] Preprocess the primary materials dataset in `ML_Practice/`
- [ ] Start reading and summarize the paper advisor gave me [Quantum and Hybrid Machine-Learning Models for
Materials-Science Tasks]([text](http://localhost:8000/2026_QML_materials.pdf))
---

##  Daily Log

###  Monday, 13 July 2026
* **Focus:** Rebuilding momentum & Quantum Mechanics fundamentals.

####  What I Learned
* **Qubit States & Notation:** Refreshed Dirac notation. A single qubit state $|\psi\rangle$ lives in a 2D complex Hilbert space, represented as:
  $$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$
  where $\alpha, \beta \in \mathbb{C}$ are complex probability amplitudes satisfying the normalization condition:
  $$|\alpha|^2 + |\beta|^2 = 1$$
* **The Bloch Sphere:** Studied the geometric representation of a single qubit where the state is mapped to a point on a unit 3D sphere using spherical coordinates:
  $$|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle$$
* **Measurement Collapse:** Measurement acts as a projection. If we measure $|\psi\rangle$ in the computational basis $\{|0\rangle, |1\rangle\}$, the probability of obtaining $|0\rangle$ is $P(0) = |\alpha|^2$, after which the state collapses completely to $|0\rangle$.

####  Problems & Roadblocks
* The concept of phase is still not fully clear. 
* I need more practice reading and interpreting quantum state notation without hesitation.

####  Tomorrow's Plan
-  Complete remaining Module 1 lectures.
-  Implement a basic 1-qubit visualization in Qiskit to see the Bloch Sphere in action.
---

###  Tuesday, 14 July 2026
* **Focus:** Mathematical frameworks of single-qubit gates and state evolution.

####  What I Learned
* **Unitary Evolution:** Quantum gates are linear operators represented by unitary matrices $U$, meaning they preserve the total probability. Mathematically:
  $$U^\dagger U = UU^\dagger = I$$
  where $U^\dagger$ is the conjugate transpose (adjoint) of $U$. This property ensures that quantum operations are inherently **reversible**.
* **Primary Single-Qubit Gates:**
  * **Hadamard ($H$):** Creates equal superposition: $H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$. Matrix form:
    $$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
  * **Pauli Gates ($X, Y, Z$):** Act as rotations around the Cartesian axes of the Bloch Sphere. For instance, the Bit-Flip gate ($X$) maps:
    $$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$
  * **Parametric Rotation Gates ($R_X, R_Y, R_Z$):** Unlike the Pauli gates ($X,Y,Z$) which perform fixed $180^\circ$ ($\pi$) flips, rotation gates accept a continuous parameter $\theta$ to rotate a qubit by an arbitrary angle around a specific axis on the Bloch Sphere. Their matrix representations are derived from the matrix exponential of the Pauli matrices:
  $$R_X(\theta) = \exp\left(-i\frac{\theta}{2}X\right) = \begin{pmatrix} \cos\left(\frac{\theta}{2}\right) & -i\sin\left(\frac{\theta}{2}\right) \\ -i\sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right) \end{pmatrix}$$
  $$R_Y(\theta) = \exp\left(-i\frac{\theta}{2}Y\right) = \begin{pmatrix} \cos\left(\frac{\theta}{2}\right) & -\sin\left(\frac{\theta}{2}\right) \\ \sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right) \end{pmatrix}$$
  $$R_Z(\theta) = \exp\left(-i\frac{\theta}{2}Z\right) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

* **Quantum Data Encoding (Angle Encoding):** Discovered the bridge between classical data and quantum states. By setting the parameter $\theta$ of a rotation gate equal to a classical normalized feature value $x_i$, we map a classical data point directly to a physical coordinate on the Bloch Sphere.
  - *Example:* Applying $R_Y(x_i)|0\rangle$ encodes the value $x_i$ into the state vector amplitudes without introducing complex phases.

####  Problems & Roadblocks
* Connecting matrix operations to the physical intuition of rotations on the Bloch Sphere requires conscious mental effort.

####  Tomorrow's Plan
- Complete the remaining lecture block in Module 2.
- Map out the matrix transformations for $Y$ and $Z$ gates manually to build muscle memory.
---

###  Wednesday, 15 July 2026
* **Focus:** Multi-qubit gates, circuit synthesis, and initial Qiskit framework deployment.

####  What I Learned
* **The CNOT Gate (Controlled-NOT):** A fundamental 2-qubit entangling gate. It applies an $X$ gate to the target qubit if and only if the control qubit is in the $|1\rangle$ state. In the computational basis $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$, its matrix representation is:
  $$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$
* Learned how the Statevector Simulator represents the complete quantum state without performing a measurement.
* Installed and became familiar with the basic structure of Qiskit.
* Learned how to import Qiskit modules, create quantum and classical registers, and construct simple quantum circuits.

####  Problems & Roadblocks
* I am still getting used to Qiskit's syntax and library structure.
* Interpreting the output of the Statevector Simulator requires more practice, especially when reading complex amplitudes.

#### Progress towards Dissertation
- Today I Transitioned from Quantum theory to writing Quantum programs on Qiskit
- Now I won't be completely clueless when reading algorithms on QML papers.
####  Tomorrow's Plan
-  Advance to the next lecture block in Module 3.
-  Code a 2-qubit circuit manually to create the Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ and print its statevector.
---

###  Thursday, 16 July 2026
* **Focus:** Circuit execution frameworks, sampling statistics, Parameterized Quantum Circuits, Observables, Pauli Operators

#### What I Learned
* Understood the general workflow of creating a circuit, running it on a simulator, and interpreting the results.
* Recognized the importance of running circuits multiple times to estimate quantum probabilities.
* Learned that experimenting with different gate combinations is an effective way to build intuition about quantum computation.
* Strengthened my ability to build quantum circuits by combining multiple quantum gates in a logical sequence.
  
####  Problems & Roadblocks
* I need more practice writing Qiskit programs without referring to previous examples.

####  Tomorrow's Plan
* Complete the last bit of theory and start reading the Research Paper provided by my advisor.
---

### Friday, 17 July 2026
* **Focus:** The Estimator Primitive, Setting up the python environment for reproducing the code. 

#### What I Learned
* Began preparing the research environment for python 
* Tried install versions mentioned in the paper as close as possible
* Understood that there are two versions of code, one which was published on the paper and one which they updated after an year.
* Realized that package compatiblity was as important as writing code  

####  Problems & Roadblocks
* I got stuck due to version incompatibility, as the older qiskit version mentioned couldn't keepup with the newer sklearn 
 












##  Weekly Summary & Next Steps
* **Key Achievements:** [What worked]
* **Gaps Identified:** [What didn't work / What I need to learn]
* **Pivot for Next Week:** [Adjustments]

