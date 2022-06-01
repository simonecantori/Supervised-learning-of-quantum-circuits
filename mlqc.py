import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, GlobalMaxPooling2D, Activation
from keras.initializers import HeNormal
from tensorflow.keras.utils import get_custom_objects
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def DataLoad(N, P, first = False):

    """
    DataLoad returns the input to the neural network and the target values.
    The input is written via one-hot encoding.
    The target values are the expectation values in reverse order (i.e. the last component
    is the expectation value of the first qubit), because this is the default mode 
    in Qiskit.
    
    The parameters are the number of qubits N and the number of gates per qubit P.
    You can choose to get only the expectation value of the first qubit by using 
    first = True (default first = False)
    """

    inp = []                 
    with open(f'Dataset/input_N{N}_P{P}.dat') as line:
        for values in line:
            try:
                gates = values.split()[0]
                inp.append(float(gates))
            except:
                continue
    
    inp = np.array(inp, dtype=int)
    
    outputs = []
    listemp = []
    with open(f'Dataset/output_N{N}_P{P}.dat') as line:
        for values in line:
            try:
                o = values.split()[0]
                listemp.append(float(o))
            except:
                outputs.append(listemp)
                listemp = []
                continue
    
    outputs=np.array(outputs)
    
    
    inp.shape = (len(inp)//(N*P), N*P)
    
    inp, index = np.unique(inp, axis = 0, return_index = True)
    outputs = outputs[index]
    
    inp.shape = (len(inp) * (N*P))
    
    
    inputs = np.zeros((len(inp), 4))
    inputs[np.arange(len(inp)), inp] = 1.
    inputs.shape = (int(len(inp) / (N*P)), N, P, 4)
    
    if first:
        outputs = outputs[:,-1]
    
    return inputs, outputs


def CnnModel(num_outputs):
    """
    CnnModel returns the convolutional neural network model
    
    num_outputs is needed to specify whether the network should predict the expectation
    values of all qubits or of the first qubit
    """
    init = HeNormal()

    class Mish(Activation):

        def __init__(self, activation, **kwargs):
            super(Mish, self).__init__(activation, **kwargs)
            self.__name__ = 'Mish'


    def mish(inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    get_custom_objects().update({'Mish': Mish(mish)})


    cnn = Sequential()
    cnn.add(Conv2D(32, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init, input_shape=(None,None,4))) 
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish'))    

    cnn.add(Conv2D(32, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(64, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(64, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(128, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(128, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(128, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(256, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(256, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(Conv2D(256, (3,3), strides = (1,1),  padding = 'same', kernel_initializer=init,))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish')) 

    cnn.add(GlobalMaxPooling2D())

    cnn.add(Dense(units=300,kernel_initializer=init)) 
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish'))   

    cnn.add(Dense(units=100,kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish'))

    cnn.add(Dense(units=50,kernel_initializer=init))     
    cnn.add(BatchNormalization())
    cnn.add(Activation('Mish'))


    cnn.add(Dense(num_outputs))
    cnn.add(BatchNormalization())
    cnn.add(Activation('sigmoid'))                                                        


    cnn.compile(
                loss='binary_crossentropy',   
                           optimizer='adam',
                          metrics=['mae', r2])
    
    return cnn

def r2(y_true, y_pred):
    """
    r2 returns the coefficient of determination R^2
    
    The parameters are the target values y_true and the values predicted by the neural 
    network y_pred
    """
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/SS_tot )


if __name__ == '__main__':
    
    N = 3   # Number of qubits
    P = 5   # Number of gates per qubit
    
    
    # first = True if we predict the first expectation value.
    # first = False if we predict N expectation values.
    first = False   
                    
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
    cnn.fit(X_train, y_train, validation_split = 0.005, epochs = 50, verbose=1,
            batch_size = 512, callbacks = [callback])    
        
    # Saving the weights
    if first:
        cnn.save_weights(f'Weights/N{N}_P{P}__firstqubit.h5')
    else:
        cnn.save_weights(f'Weights/N{N}_P{P}__allqubits.h5')
    
    # Predictions
    y_pred_test = cnn.predict(X_test)
    
    # If we predict N target values, we have outputs of shape (len(y_test), N).
    # We reshape these arrays into 1-D arrays, then we calculate the R^2 on these new values.
    if not first:
        y_pred_test.shape = len(y_pred_test)*N
        y_test.shape = len(y_test)*N
        
        
    print(r2_score(y_test, y_pred_test))
