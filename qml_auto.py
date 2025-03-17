import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import Estimator
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
import sys

env_vars_file = sys.argv[1]
env_vars = {}
with open(env_vars_file, 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        env_vars[key] = value.strip("'")

jobid = int(env_vars.get('LSB_REMOTEJID'))
num_qubits = 10
split = int(env_vars.get('split')) #Select split for training and testing data
path = '../data'
dataset = str(env_vars.get('dataset')) #Select dataset
excel_file = f'{path}/{dataset}.xlsx'
df = pd.read_excel(excel_file, sheet_name='normalised_data_0_5')
classes_string = 'Phase (373K)'
classes = df[classes_string].to_numpy()
smiles = df['SMILES'].to_numpy()
property_string = 'Boiling Point'
property = df[property_string].to_numpy()
df_train_indices = pd.read_excel(excel_file, sheet_name='train_split', header=None).to_numpy()
df_train_split = df_train_indices[:,split-1] - 1
df_train_smiles = smiles[df_train_split]
df_train_classes = classes[df_train_split]
df_test_indices = pd.read_excel(excel_file, sheet_name='test_split', header=None).to_numpy()
df_test_split = df_test_indices[:,split-1] - 1
df_test_smiles = smiles[df_test_split]
df_test_classes = classes[df_test_split]
df_train_property = property[df_train_split]
df_test_property = property[df_test_split]
num_train_points = df_train_classes.shape[0]
num_test_points = df_test_classes.shape[0]
for i in range(num_train_points):
    if df_train_classes[i] == 0:
        df_train_classes[i] = -1
for i in range(num_test_points):
    if df_test_classes[i] == 0:
        df_test_classes[i] = -1

train_circuits = []
test_circuits = []
initial_layer = 'ry'
entangling_layer = 'rxx'
feature_layers = 1
atom_factor = 3 # 2.4 is default, however 3 is better in terms of loss 

for i in range(num_train_points):
    cm_train = coulomb_matrix(df_train_smiles[i], add_hydrogens=False, atom_factor=atom_factor)
    circuit_train = matrix_to_circuit(cm_train, num_qubits, n_layers=feature_layers, initial_layer=initial_layer, entangling_layer=entangling_layer)
    train_circuits.append(circuit_train)
for i in range(num_test_points):
    cm_test = coulomb_matrix(df_test_smiles[i], add_hydrogens=False, atom_factor=atom_factor)
    circuit_test = matrix_to_circuit(cm_test, num_qubits, n_layers=feature_layers, initial_layer=initial_layer, entangling_layer=entangling_layer)
    test_circuits.append(circuit_test)

#Variational quantum classifier
ansatz_layers = int(env_vars.get('ansatz_layers')) #Number of ansatz layers
maxiter = int(env_vars.get('maxiter')) #Number of iterations for the optimizer
operator = Pauli(str(env_vars.get('operator'))) #Operator for the cost function
two_local_initial_layer = str(env_vars.get('two_local_initial_layer')) #Initial layer for the 2-local ansatz
two_local_entangling_layer = str(env_vars.get('two_local_entangling_layer')) #Entangling layer for the 2-local ansatz
ansatz_entanglement = str(env_vars.get('ansatz_entanglement')) #Entanglement for the ansatz
# bh_iterations = 5 #Number of iterations for basin-hopping, basin-hopping happens every maxiter/bh_iterations steps
# max_runs = maxiter / bh_iterations
# bh_stepsize = 0.1 #Stepsize for basin-hopping
# bh_temp = 1 #Temperature for basin-hopping, if 0 then monotonic basin-hopping is carried out
ansatz = TwoLocal(num_qubits, two_local_initial_layer, two_local_entangling_layer, reps=ansatz_layers, entanglement=ansatz_entanglement, skip_final_rotation_layer=True)

backend_options = {
    "method": "statevector",
    "device": "CPU",
    "max_parallel_threads": 0,
    "max_parallel_experiments": 0,
    "max_parallel_shots": 0,
    "statevector_parallel_threshold": 2
}
run_options = {
    "shots": None,
    "approximation": True
}
#estimator = Estimator(options=dict(backend_options=backend_options))
estimator = Estimator(backend_options=backend_options, run_options=run_options)
estimator_qnn = EstimatorQNN(estimator=estimator, circuit=ansatz, input_params=None, observables=operator)



#estimator_qnn = EstimatorQNN(circuit=ansatz, input_params=None, observables=operator)

def callback(weights, obj_func_eval):
    weight_vals.append(weights)
    objective_vals.append(obj_func_eval)

num_points = int(env_vars.get('num_points')) #Number of points to train
training_scores = []
test_scores = []

json_dict = {'num_points': num_points, 'split': split, 'dataset': dataset, 'property': classes_string,
             'initial_layer': initial_layer, 'entangling_layer': entangling_layer, 
             'two_local_initial_layer': two_local_initial_layer, 
             'two_local_entangling_layer': two_local_entangling_layer, 'ansatz_entanglement': ansatz_entanglement, 
             'ansatz_layers': ansatz_layers, 'maxiter': maxiter, 
             'operator': str(operator)}

newpath = f'classifier_models/{jobid}/{initial_layer}_{entangling_layer}/{two_local_initial_layer}_{two_local_entangling_layer}_{ansatz_entanglement}/split{split}/a{ansatz_layers}_m{int(maxiter/1000)}k'
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(json_dict, flush=True)
write_json(f'{newpath}', 'data.json', json_dict)

for i in range(num_points):
    objective_vals = []
    weight_vals = []
    initial_point = (np.random.random(ansatz.num_parameters) - 0.5) * 2 * np.pi 
    estimator_classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=maxiter), callback=callback, initial_point=initial_point)

    estimator_classifier.fit(train_circuits, df_train_classes)
    objective_vals = np.array(objective_vals)
    weight_vals = np.array(weight_vals)
    training_scores.append(estimator_classifier.score(train_circuits, df_train_classes))
    test_scores.append(estimator_classifier.score(test_circuits, df_test_classes))

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
