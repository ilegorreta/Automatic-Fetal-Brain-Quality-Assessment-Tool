# Automatic Fetal Brain Quality Assessment
## Developed by Iván Legorreta
### Contact info: ilegorreta@outlook.com

---

The aim of this project was to develop a Quality Assessment tool for fetal brain MRIs, which is able to score each volume through a deep learning regression model. Developed using Python3 and Keras/Tensorflow framework.

Our network architecture consists of a non-linear configuration, known as Residual Network (ResNet) architecture: 
![Resnet Architecture Diagram](https://github.com/ilegorreta/Automatic-Fetal-Brain-Quality-Assessment-Tool/blob/main/resnet_architecture_diagram.png)

Given that we are dealing with an unbalanced distribution regarding input dataset, we applied different weights to each input class to compensate for the imbalance in the training sample.

---
### Requirements
* Linux environment
* Python3
* Conda/Anaconda setup

### Installation
1. Create the environment from the conda_environment.yml file:
```python
  conda env create -f conda_environment.yml
```
The first line of the yml file sets the new environment's name.

1.  Activate the new environment: conda activate myenv

1. Verify that the new environment was installed correctly:
```python
  conda env list
```
You can also use conda info --envs

For more information, visit [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).
