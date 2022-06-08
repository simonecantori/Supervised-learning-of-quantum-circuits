# ML_QC
Relevant scripts and data for the paper entitled "Supervised learning of random quantum circuits via scalable neural networks"

## Table of contents
* [Python scripts](#python-scripts)
* ["Dataset" Folder]("Dataset"-Folder)
* ["Weights" Folder]("Weights"-Folder)

## Python scripts
- "mlqc.py" contains all the important functions. The "main" part of this script can be used to train and test the neural network on quantum circuits of different sizes.

- "Extrapolation_train_test.py" is used to train a neural network on quantum circuit of a certain size. We can use this neural network to predict expectation values of quantum circuits of different sizes.

- "Extrapolation_test.py" is used to make prediction on quantum circuits given a pre-train model.

- "circ_gen.py" generates random quantum circuits with the gates belonging to the set [T, H, CX].
 
## "Dataset" Folder
It contains some of the data used to get the results shown in the paper.
One can generate more data by using "circ_gen.py".
Input data contains integers in the range [0, 3]. Each integer is related to a quantum gate. In the python scripts, we use the one-hot encoding technique to convert each integer to an array.

	
## "Weights" Folder
This folder can be used to store the weights of the neural network that we want to use again.
We have already uploaded a model of a neural network trained on quantum circuits with 10 qubit and 6 gates per qubit. This neural network has been trained using an "EarlyStopping" with "patience = 5". One can decide to use different training protocols.
