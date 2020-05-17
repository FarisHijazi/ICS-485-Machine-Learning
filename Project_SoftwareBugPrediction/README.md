# ICS 485 : Machine Learning - Task3: Neural Networks

## Term Project: Software Bug Prediction

This is the neural networks section of the project.

**Team member:** Faris Hijazi s201578750

There are 2 solution files:
- [Project_SoftwareBugPrediction\proj-task1-2.ipynb]() (solved by other team members)
- [Project_SoftwareBugPrediction\proj-task3-NeuralNetworks.ipynb]() which is my part (Faris Hijazi)

**Abstract:** Develop advanced machine learning-based classification models to
satisfactorily classify **software bugs** (binary and multiclass) for data collected at the
University of Geneva in Switzerland.

**Dataset:** The dataset is related to the bug prediction dataset. It contains data about the
following software systems:
- Eclipse JDT Core
- Eclipse PDE UI
- Equinox Framework
- Lucene
- Mylyn

## Problem statement

Find it in the [Project_SoftwareBugPrediction\ICS485-Project.pdf]() file.

---

**Training/Validation/Testing Set:**
Divide the dataset into Training/Validation/Testing Set by randomly distributing 70%
for training, 15% for validation, and 15% for the test set.

**Task 1 :**
Provide **3 different binary classifiers** to predict the bug given the software metrics
considered. In case of more than one bug, you should treat the sample as infected with a
bug (class 1). Investigate the following issues:
a. Classifier optimization and hyper-parameter tuning.
b. The metrics to measure the classifier performance.
c. Make sure to run your models on the testing data.


**Task 2 :**
Provide **3 different multiclassifiers** to predict the bugs ( **0, 1, and 2** ) given the software
metrics considered. In case of more than one bug, you should treat the sample as infected
with more than 2 bugs be considered as class 2. Investigate the following issues:
a. Classifier optimization and hyper-parameter tuning.
b. The metrics to measure the classifier performance.
c. Make sure to run your models the testing data.

**Task 3 :**
Train a **feedforward neural network** to predict the bugs for the data provided. Determine
the optimal training parameters for your neural network that are sufficiently general to
predict the bugs on any withheld data that you will not have for testing purposes. You
will, therefore, need to devise and execute a plan that uses the given data for training and
testing in a manner that most closely mimics the real test (on withheld data) including:

a. Number of hidden layers  
b. Hidden layer nodes  
c. Training function  
d. Learning function  
e. Iterations (epochs)  
