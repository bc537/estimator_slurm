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

num_qubits = 10
split = int(os.environ.get('split')) #Select split for training and testing data
path = '../data'
dataset = str(os.environ.get('dataset')) #Select dataset
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
atom_factor = 3 

for i in range(num_train_points):
    cm_train = coulomb_matrix(df_train_smiles[i], add_hydrogens=False, atom_factor=atom_factor)
    circuit_train = matrix_to_circuit(cm_train, num_qubits, n_layers=feature_layers, initial_layer=initial_layer, entangling_layer=entangling_layer)
    train_circuits.append(circuit_train)
for i in range(num_test_points):
    cm_test = coulomb_matrix(df_test_smiles[i], add_hydrogens=False, atom_factor=atom_factor)
    circuit_test = matrix_to_circuit(cm_test, num_qubits, n_layers=feature_layers, initial_layer=initial_layer, entangling_layer=entangling_layer)
    test_circuits.append(circuit_test)

#Variational quantum classifier
ansatz_layers = int(os.environ.get('ansatz_layers')) #Number of ansatz layers
maxiter = int(os.environ.get('maxiter')) #Number of iterations for the optimizer
operator = Pauli(str(os.environ.get('operator'))) #Operator for the cost function
two_local_initial_layer = str(os.environ.get('two_local_initial_layer')) #Initial layer for the 2-local ansatz
two_local_entangling_layer = str(os.environ.get('two_local_entangling_layer')) #Entangling layer for the 2-local ansatz
ansatz_entanglement = str(os.environ.get('ansatz_entanglement')) #Entanglement for the ansatz
bh_iterations = int(os.environ.get('bh_iterations')) #Number of iterations for basin-hopping, basin-hopping happens every maxiter/bh_iterations steps
max_runs = maxiter / bh_iterations
bh_stepsize = float(os.environ.get('bh_iterations')) #Stepsize for basin-hopping
bh_temp = float(os.environ.get('bh_temp')) #Temperature for basin-hopping, if 0 then monotonic basin-hopping is carried out
ansatz = TwoLocal(num_qubits, two_local_initial_layer, two_local_entangling_layer, reps=ansatz_layers, entanglement=ansatz_entanglement, skip_final_rotation_layer=True)
estimator_qnn = EstimatorQNN(circuit=ansatz, input_params=None, observables=operator)

def callback(weights, obj_func_eval):
    weight_vals.append(weights)
    objective_vals.append(obj_func_eval)

def bh_callback(x, f, accept):
    x_total.append(x)
    f_total.append(f)
    accept_total.append(accept)

num_points = int(os.environ.get('num_points')) #Number of points to train
training_scores = []
test_scores = []

json_dict = {'num_points': num_points, 'split': split, 'dataset': dataset, 'property': property_string,
             'initial_layer': initial_layer, 'entangling_layer': entangling_layer, 
             'two_local_initial_layer': two_local_initial_layer, 
             'two_local_entangling_layer': two_local_entangling_layer, 'ansatz_entanglement': ansatz_entanglement, 
             'ansatz_layers': ansatz_layers, 'maxiter': maxiter, 
             'operator': str(operator), 'bh_iterations': bh_iterations, 'bh_stepsize': bh_stepsize, 'bh_temp': bh_temp}

newpath = f'regression_bh_models/{initial_layer}_{entangling_layer}/{two_local_initial_layer}_{two_local_entangling_layer}_{ansatz_entanglement}/split{split}/a{ansatz_layers}_m{int(maxiter/1000)}k'
if not os.path.exists(newpath):
    os.makedirs(newpath)

print(json_dict, flush=True)
write_json(f'{newpath}', 'data.json', json_dict)

for i in range(num_points):
    objective_vals = []
    weight_vals = []
    x_total = []
    f_total = []
    accept_total = []
    initial_point = (np.random.random(ansatz.num_parameters) - 0.5) * 2 * np.pi 
    opt = BasinHopping(
    minimizer_kwargs={"method": 'COBYLA', "jac": False, "options": {"maxiter": max_runs}},
    options=dict(niter=bh_iterations - 1, stepsize=bh_stepsize, callback=bh_callback, T=bh_temp))
    estimator_classifier = NeuralNetworkRegressor(estimator_qnn, optimizer=opt, callback=callback, initial_point=initial_point)

    estimator_classifier.fit(train_circuits, df_train_property)
    objective_vals = np.array(objective_vals)
    weight_vals = np.array(weight_vals)

    min_ind = f_total.index(min(f_total))
    min_x = x_total[min_ind]    
    est = NeuralNetworkRegressor(estimator_qnn, optimizer=COBYLA(maxiter=1), initial_point=min_x)
    est.fit(train_circuits, df_train_classes)
    training_scores.append(est.score(train_circuits, df_train_property))
    test_scores.append(est.score(test_circuits, df_test_property))

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