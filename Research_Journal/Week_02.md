# Week 2: [Dates, June 20 - June 27]

##  Weekly Goals
- Completely reproduce the codes in the given research paper and learn it to the point of explaining it to another person
- Read and Understand the paper


##  Daily Log

###  Monday, 20 July 2026
* **Focus:** Reproduce the codes from QSVC and QSVR files in the QMML_SFE_Mg repository

#### What I Learned
* Make sure every library share the same timeline.
* QSVC and QSVR share the same overall pipeline, with QSVR extending the method to regression through the epsilon parameter.
* It takes an enormus time for the codes on the repository to run, and for checking whether the code is running or not, its better to reduce the number of parameters or value of it.
* The smallest code in the repository will take 22 hours to run.

####  Problems & Roadblocks
* It takes a lot of time for the code to be run.
* The paper didn't mention about compatible versions other that qiskit.
* Got stuck during the installation of numpy and pandas, as it was newer versions which were incompatible here.

####  Tomorrow's Plan
* Try to reproduce QVC and QVR to jupyter notebook.

###  Tuesday, 21 July 2026
* **Focus:** Reproduce VQC,VQR and Understanding and making notes for all the reproduced codes.

### What I Learned
* Understood the purpose of every major library imported in the QSVC implementation and how classical Python libraries (NumPy, Pandas, Scikit-learn) integrate with Qiskit's Quantum Machine Learning framework.
* Learned how the QSVC workflow combines classical machine learning (data preprocessing, cross-validation, evaluation, and optimization) with quantum components (feature maps and quantum kernels).
* Gained a clear understanding of the role of the ZZFeatureMap in encoding classical material properties into quantum states before classification.
* Learned how the Fidelity Quantum Kernel computes similarity between encoded quantum states and replaces the classical kernel used in Support Vector Machines.
* Understood that QSVC remains a classical Support Vector Machine, with the quantum advantage coming from the quantum kernel rather than the classifier itself.
* Studied how the dataset is prepared through train-test splitting, feature scaling, and preprocessing before being passed to the quantum machine learning model.
* Learned the importance of reproducibility through practices such as fixing the random seed, matching software versions, and preserving the original implementation.
* Understood how hyperparameters—including feature map repetitions (reps), entanglement topology, and the regularization parameter (C)—can influence the performance of a Quantum Support Vector Classifier.
* Followed the complete training pipeline from preprocessing the dataset to constructing the QSVC model, training it, and generating predictions on unseen data.
* Began analyzing the code from a research perspective by questioning the authors' implementation choices and identifying potential areas for future improvement, rather than simply reproducing the code.

####  Problems & Roadblocks
* It takes time to fully understand the code. For fully analysing the code and understand how it came to be , hours are spent.
* Probably would need to spent a few days fully reproducing and understanding the code.

#### Today's Biggest Realization
A Quantum Support Vector Classifier is still fundamentally a classical Support Vector Machine—the quantum component lies in how similarities between data points are computed using quantum states and kernels.

####  Tomorrow's Plan
* Try finish and analyse QSVC and QSVR and start VQC.
* update the journal

###  Wednesday, 22 July 2026
* **Focus:** Completely analyse and write detailed notes about QSVC and QSVR . Rearrange my Git Repository.
  Detailed notes are in Dissertation_Masters_QML/Documentation/01_QSVC.md
  
### What I Learned
* Developed a clearer understanding of the complete workflow of both QSVC and QSVR, from data preprocessing and feature scaling to model construction, training, prediction, and evaluation.
* Understood how classical machine learning components (cross-validation, preprocessing, SVM optimization) are integrated with quantum components (feature maps and quantum kernels) to form hybrid quantum machine learning models.
* Learned the similarities and differences between Quantum Support Vector Classification (QSVC) and Quantum Support Vector Regression (QSVR), particularly how they address classification and regression tasks while sharing the same quantum kernel framework.
* Reinforced the importance of reproducibility by studying the implementation in detail rather than treating it as a black box.
* Improved my ability to read, interpret, and document research code systematically, making it easier to revisit and modify the implementation in the future.
* Began thinking critically about the authors' design choices and identifying areas that could potentially be explored or improved during my dissertation work.

####  Problems & Roadblocks
* It takes an extremly large amount of my time and focus to make a detailed note.
* Don't know if I will be able to completely reproduce the code at the end of the month.
  
#### Today's Biggest Realization
Successfully reproducing a research paper is only the first step; truly understanding the purpose of every function, parameter, and design choice is what prepares me to develop and contribute my own research.

####  Tomorrow's Plan
* Complete transferring hybrid_QNNC, hybrid_QNNR .
* Compile notes for QSVR and find a way to update github.