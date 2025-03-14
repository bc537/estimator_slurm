import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B, SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_machine_learning.optimizers import BasinHopping
from bond_encoding.bond import coulomb_matrix, matrix_to_circuit, write_json
import os

bit_dim = 11
num_qubits = 10
split = int(os.environ.get('split')) #Select split for training and testing data
path = f'../data/angle_split/split{split}'
df_train_classes = pd.read_excel(f'{path}/qc_hc_train_classes_{bit_dim}q{num_qubits}.xlsx', header=None).to_numpy().flatten()
df_train_angle = pd.read_excel(f'{path}/qc_hc_train_input_{bit_dim}q{num_qubits}.xlsx', header=None).to_numpy()
df_test_classes = pd.read_excel(f'{path}/qc_hc_test_classes_{bit_dim}q{num_qubits}.xlsx', header=None).to_numpy().flatten()
df_test_angle = pd.read_excel(f'{path}/qc_hc_test_input_{bit_dim}q{num_qubits}.xlsx', header=None).to_numpy()
num_inputs = df_train_angle.shape[1]
num_train_points = df_train_classes.shape[0]
num_test_points = df_test_classes.shape[0]
for i in range(num_train_points):
    if df_train_classes[i] == 0:
        df_train_classes[i] = -1
for i in range(num_test_points):
    if df_test_classes[i] == 0:
        df_test_classes[i] = -1

#Variational quantum classifier
feature_layers = int(os.environ.get('feature_layers')) #Number of feature layers
ansatz_layers = int(os.environ.get('ansatz_layers')) #Number of ansatz layers
maxiter = int(os.environ.get('maxiter')) #Number of iterations for the optimizer
operator = Pauli(str(os.environ.get('operator'))) #Operator for the cost function
two_local_initial_layer = str(os.environ.get('two_local_initial_layer')) #Initial layer for the 2-local ansatz
two_local_entangling_layer = str(os.environ.get('two_local_entangling_layer')) #Entangling layer for the 2-local ansatz
ansatz_entanglement = str(os.environ.get('ansatz_entanglement')) #Entanglement for the ansatz
feature_map = RealAmplitudes(num_qubits=num_qubits, entanglement="linear", parameter_prefix='x', reps=feature_layers, skip_final_rotation_layer=True)
ansatz = TwoLocal(num_qubits, two_local_initial_layer, two_local_entangling_layer, reps=ansatz_layers, entanglement=ansatz_entanglement, skip_final_rotation_layer=True)
qc = QNNCircuit(num_qubits=num_qubits, feature_map=feature_map, ansatz=ansatz)
estimator_qnn = EstimatorQNN(circuit=qc, observables=operator)

def callback(weights, obj_func_eval):
    weight_vals.append(weights)
    objective_vals.append(obj_func_eval)

num_points = int(os.environ.get('num_points')) #Number of points to train
training_scores = []
test_scores = []

json_dict = {'num_points': num_points, 'split': split, 'feature_layers': feature_layers,
             'two_local_initial_layer': two_local_initial_layer, 
             'two_local_entangling_layer': two_local_entangling_layer, 'ansatz_entanglement': ansatz_entanglement, 
             'ansatz_layers': ansatz_layers, 'maxiter': maxiter, 
             'operator': str(operator)}

newpath = f'classifier_models/angle_encoding/{two_local_initial_layer}_{two_local_entangling_layer}_{ansatz_entanglement}/split{split}/a{ansatz_layers}_m{int(maxiter/1000)}k'
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(json_dict, flush=True)
write_json(f'{newpath}', 'data.json', json_dict)

for i in range(num_points):
    objective_vals = []
    weight_vals = []
    initial_point = (np.random.random(ansatz.num_parameters) - 0.5) * 2 * np.pi 
    estimator_classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=maxiter), callback=callback, initial_point=initial_point)

    estimator_classifier.fit(df_train_angle, df_train_classes)
    objective_vals = np.array(objective_vals)
    weight_vals = np.array(weight_vals)
    training_scores.append(estimator_classifier.score(df_train_angle, df_train_classes))
    test_scores.append(estimator_classifier.score(df_test_angle, df_test_classes))

    newpath2 = f'{newpath}/p{i+1}'
    if not os.path.exists(newpath2):
        os.makedirs(newpath2)

    estimator_classifier.save(f'{newpath2}/qml')
    np.savetxt(f"{newpath2}/objective_vals", objective_vals)
    np.savetxt(f"{newpath2}/weight_vals", weight_vals)

    training_scores_save = np.array(training_scores)
    test_scores_save = np.array(test_scores)

    np.savetxt(f"{newpath}/training_scores", training_scores_save)
    np.savetxt(f"{newpath}/test_scores", test_scores_save)