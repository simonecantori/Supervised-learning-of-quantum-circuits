"""
We use a saved model to predict different quantum circuits
"""

from mlqc import DataLoad, CnnModel
from sklearn.metrics import r2_score

N = 10   # Number of qubits for the saved model
P = 6   # Number of gates per qubit for the saved model


# first = True if we predict the first expectation value.
# first = False if we predict N expectation values.
first = True 
                

# Defining the model
if first:
    cnn = CnnModel(1)
else:
    cnn = CnnModel(N)

cnn.summary()   

# Loading the weights
cnn.load_weights(f'Weights/N{N}_P{P}.h5')      

N = 11   # Number of qubits of quantum circuits that we want to study with the saved model
P = 6   # Number of gates per qubit

# Loading the instances that we want to study
X_test, y_test = DataLoad(N, P, first = first)       


# Predictions
y_pred_test = cnn.predict(X_test)

# If we predict N target values, we have outputs of shape (len(y_test), N).
# We reshape these arrays into 1-D arrays, then we calculate the R^2 on these new values.
if not first:
    y_pred_test.shape = len(y_pred_test)*N
    y_test.shape = len(y_test)*N
    
    
# Testing on quantum circuits with N = 11, 12, 15, 20 and P = 6.
for N in [11, 12, 15, 20]:
    P = 6
    
    inputs, outputs = DataLoad(N, P, first = first)
    
    y_pred_test = cnn.predict(inputs)
    y_test = outputs
        
        
    print(f'R^2 for quantum circuits with N = {N} qubits is {r2_score(y_test, y_pred_test)}')