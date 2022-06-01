"""
Train on quantum circuit with N = 10 and P = 6.

We test the model on quantum circuit with N = 11, 12, 15, 20 and P = 6 (extrapolation).
"""

from mlqc import DataLoad, CnnModel
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

N = 10   # Number of qubits
P = 6   # Number of gates per qubit


# first = True if we predict the first expectation value.
# first = False if we predict N expectation values.
first = True 
                
# Loading inputs and outputs of the neural network
inputs, outputs = DataLoad(N, P, first = first)

# Defining the model
if first:
    cnn = CnnModel(1)
else:
    cnn = CnnModel(N)

cnn.summary()         

                   
# Split in train and test set
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 2000) 

del(inputs)
del(outputs)

# After 5 epochs (customizable) without any improvement (for the validation loss) 
# the training stops and we get the best model weights    
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights = True)

# Training the model 
cnn.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 50, verbose=1,
        batch_size = 512, callbacks = [callback])    
    
# Saving the weights
cnn.save_weights(f'Weights/N{N}_P{P}.h5')

# Predictions
y_pred_test = cnn.predict(X_test)

# If we predict N target values, we have outputs of shape (len(y_test), N).
# We reshape these arrays into 1-D arrays, then we calculate the R^2 on these new values.
if not first:
    y_pred_test.shape = len(y_pred_test)*N
    y_test.shape = len(y_test)*N
    
    
print(r2_score(y_test, y_pred_test))

# Testing on quantum circuits with N = 11, 12, 15, 20 and P = 6.
for N in [11, 12, 15, 20]:
    P = 6
    
    inputs, outputs = DataLoad(N, P, first = first)
    
    y_pred_test = cnn.predict(inputs)
    y_test = outputs
        
        
    print(f'R^2 for quantum circuits with N = {N} qubits is {r2_score(y_test, y_pred_test)}')