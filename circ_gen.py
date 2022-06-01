from qiskit import QuantumCircuit, assemble, Aer
import random
import numpy as np

sim = Aer.get_backend('aer_simulator')  
flag = 0
num_circuits = 1000000 #Number of quantum circuits that will be simulated
N = 3 # Number of qubits
P = 5 # Number of gates per qubit

filei = open(f'Dataset/input_N{N}_P{P}.dat','w')
fileo = open(f'Dataset/output_N{N}_P{P}.dat','w')

circ = np.zeros((N,P))
for n in range(num_circuits):
    
    qc = QuantumCircuit(N)  
    
    for j in range(P): # Loop over the gates layers
        col = np.zeros(N)
        countarget = 0
        for i in range(N):
            if col[i] != 3: # It is not the target of the CX-gate
                listemp = []
                if flag == 0: # There isn't a CX-gate in this layer of gates
                    r = random.randint(0, 2) # Random choice between H, T and the control of the CX-gate
                    listemp.append(r)
                    if r == 2: # If it is the control of the CX-gate
                        r2 = random.randint(1, N - 1)
                        t = (i + r2) % N # This is the target of the CX-gate
                        listemp.append(t)
                else: # If the CX-gate has been defined, only 1-qubit gate can be added in this layer of gates
                    r = random.randint(0, 1)
                    listemp.append(r)               
                col[i]=listemp[0]
                if len(listemp) == 2:
                    flag = 1 # Now there is a CX-gate in this layer of gates
                    col[listemp[1]] = 3
        for i in range(N): # We build the j-th layer of gates
            if col[i] == 0:
                qc.t(i)
            elif col[i] == 1:
                qc.h(i)
            elif col[i] == 2:
                qc.cx(i, t)
        

         # We store the gates that we have generated in a list           
        circ[:,j] = col
        flag = 0
        
    # Simulation of the quantum circuit
    qc.save_statevector()
    qobj = assemble(qc)
    result = sim.run(qobj).result()
    
    
    valcount = np.zeros((N)) # We store the expectation values here
    
    key = np.array(list(result.get_counts().keys())) # output bit strings
    val = np.array(list(result.get_counts().values())) # output bit strings probabilities
    
    # Calculation of the rescaled expectation values
    for i in range(len(key)):
        for j in range(len(key[i])):
            if key[i][j] == '1':
                valcount[j] += val[i]
    
    # Write on files
    for i in range(N):
        for j in range(P): 
            filei.write(f'{circ[i][j]}\n')
    filei.write('\n')
    for j in valcount:
        fileo.write(f'{j}\n')
    fileo.write('\n')

        
fileo.close()
filei.close()
