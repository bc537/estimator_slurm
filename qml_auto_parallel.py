import os
import sys
import json
import time
import threading
import psutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer.primitives import Estimator
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from bond_encoding.bond import coulomb_matrix, matrix_to_circuit, write_json
from joblib import Parallel, delayed

# Global variable to store the maximum CPU usage observed.
max_cpu_usage = [0]
# Event to signal the CPU monitoring thread to stop.
stop_event = threading.Event()

def monitor_cpu_usage(interval=0.5):
    """
    Monitor CPU usage at a given interval (in seconds) and record the maximum CPU usage observed.
    """
    while not stop_event.is_set():
        # Sample CPU usage across all cores.
        usage = psutil.cpu_percent(interval=interval, percpu=True)
        max_cpu_usage[0] = max(max_cpu_usage[0], max(usage))

def load_env_vars(filepath: str) -> dict:
    """
    Load environment variables from a file.
    The file should have lines in the format: KEY='value'
    """
    env_vars = {}
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            env_vars[key] = value.strip("'")
    return env_vars

def load_dataset(excel_file: str, split: int):
    """
    Load the dataset from an Excel file and split it into training and testing sets.
    Returns:
        train_smiles, train_classes, test_smiles, test_classes,
        train_property, test_property
    """
    df = pd.read_excel(excel_file, sheet_name='normalised_data_0_5')
    classes = df['Phase (373K)'].to_numpy()
    smiles = df['SMILES'].to_numpy()
    properties = df['Boiling Point'].to_numpy()

    # Load train/test indices and adjust for 0-indexing.
    train_indices = pd.read_excel(excel_file, sheet_name='train_split', header=None).to_numpy()
    test_indices = pd.read_excel(excel_file, sheet_name='test_split', header=None).to_numpy()
    train_split = train_indices[:, split - 1] - 1
    test_split = test_indices[:, split - 1] - 1

    train_smiles = smiles[train_split]
    test_smiles = smiles[test_split]
    train_classes = classes[train_split]
    test_classes = classes[test_split]
    train_property = properties[train_split]
    test_property = properties[test_split]

    # Replace class label 0 with -1.
    train_classes = np.where(train_classes == 0, -1, train_classes)
    test_classes = np.where(test_classes == 0, -1, test_classes)

    return train_smiles, train_classes, test_smiles, test_classes, train_property, test_property

def generate_circuits(smiles_array, num_qubits: int, feature_layers: int,
                      initial_layer: str, entangling_layer: str, atom_factor: float):
    """
    Generate quantum circuits from an array of SMILES strings.
    Each SMILES string is converted to a Coulomb matrix and then to a quantum circuit.
    """
    circuits = []
    for smiles in smiles_array:
        cm = coulomb_matrix(smiles, add_hydrogens=False, atom_factor=atom_factor)
        circuit = matrix_to_circuit(cm, num_qubits, n_layers=feature_layers,
                                    initial_layer=initial_layer,
                                    entangling_layer=entangling_layer)
        circuits.append(circuit)
    return circuits

def create_ansatz(num_qubits: int, ansatz_layers: int,
                  two_local_initial_layer: str, two_local_entangling_layer: str,
                  ansatz_entanglement: str):
    """
    Create a variational ansatz using a TwoLocal circuit.
    """
    return TwoLocal(num_qubits, two_local_initial_layer, two_local_entangling_layer,
                    reps=ansatz_layers, entanglement=ansatz_entanglement,
                    skip_final_rotation_layer=True)

def train_single_model_joblib(run_index: int, train_circuits, train_labels,
                              test_circuits, test_labels, estimator_qnn, 
                              maxiter: int, base_save_path: str) -> tuple:
    """
    Train one model instance, save its outputs, and return its training and test scores.
    This function is designed to be executed in parallel using joblib.
    """
    objective_vals = []
    weight_vals = []

    def callback(weights, obj_func_eval):
        """Callback to record weights and objective function evaluations."""
        weight_vals.append(weights)
        objective_vals.append(obj_func_eval)

    # Generate a random initial point.
    initial_point = (np.random.random(estimator_qnn.circuit.num_parameters) - 0.5) * 2 * np.pi

    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=maxiter),
                                         callback=callback, initial_point=initial_point)
    
    classifier.fit(train_circuits, train_labels)
    train_score = classifier.score(train_circuits, train_labels)
    test_score = classifier.score(test_circuits, test_labels)

    # Create a subdirectory for this run.
    run_save_path = os.path.join(base_save_path, f'p{run_index+1}')
    os.makedirs(run_save_path, exist_ok=True)
    classifier.save(os.path.join(run_save_path, 'qml'))
    np.savetxt(os.path.join(run_save_path, "objective_vals.txt"), np.array(objective_vals))
    np.savetxt(os.path.join(run_save_path, "weight_vals.txt"), np.array(weight_vals))

    return run_index, train_score, test_score

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <env_vars_file>")
        sys.exit(1)
    
    # Load environment variables.
    env_vars_file = sys.argv[1]
    env_vars = load_env_vars(env_vars_file)

    # Extract configuration parameters.
    jobid = int(env_vars.get('LSB_REMOTEJID'))
    split = int(env_vars.get('split'))
    dataset = env_vars.get('dataset')
    ansatz_layers = int(env_vars.get('ansatz_layers'))
    maxiter = int(env_vars.get('maxiter'))
    operator = Pauli(env_vars.get('operator'))
    two_local_initial_layer = env_vars.get('two_local_initial_layer')
    two_local_entangling_layer = env_vars.get('two_local_entangling_layer')
    ansatz_entanglement = env_vars.get('ansatz_entanglement')
    num_points = int(env_vars.get('num_points'))

    # Fixed parameters for circuit generation.
    num_qubits = 10
    feature_layers = 1
    initial_layer = 'ry'
    entangling_layer = 'rxx'
    atom_factor = 3

    # Paths for data and saving models.
    data_path = '../data'
    excel_file = os.path.join(data_path, f'{dataset}.xlsx')

    # Load dataset and generate circuits.
    train_smiles, train_classes, test_smiles, test_classes, _, _ = load_dataset(excel_file, split)
    train_circuits = generate_circuits(train_smiles, num_qubits, feature_layers,
                                       initial_layer, entangling_layer, atom_factor)
    test_circuits = generate_circuits(test_smiles, num_qubits, feature_layers,
                                      initial_layer, entangling_layer, atom_factor)

    # Build the variational ansatz and corresponding QNN.
    ansatz = create_ansatz(num_qubits, ansatz_layers, two_local_initial_layer,
                           two_local_entangling_layer, ansatz_entanglement)
    estimator_qnn = EstimatorQNN(circuit=ansatz, input_params=None, observables=operator)

    # Prepare configuration dictionary.
    config = {
        'num_points': num_points,
        'split': split,
        'dataset': dataset,
        'property': 'Phase (373K)',
        'initial_layer': initial_layer,
        'entangling_layer': entangling_layer,
        'two_local_initial_layer': two_local_initial_layer,
        'two_local_entangling_layer': two_local_entangling_layer,
        'ansatz_entanglement': ansatz_entanglement,
        'ansatz_layers': ansatz_layers,
        'maxiter': maxiter,
        'operator': str(operator)
    }

    base_save_path = os.path.join('classifier_models', str(jobid),
                                  f'{initial_layer}_{entangling_layer}',
                                  f'{two_local_initial_layer}_{two_local_entangling_layer}_{ansatz_entanglement}',
                                  f'split{split}',
                                  f'a{ansatz_layers}_m{int(maxiter/1000)}k')
    os.makedirs(base_save_path, exist_ok=True)
    print(json.dumps(config, indent=2))
    write_json(base_save_path, 'data.json', config)

    # Start the CPU monitoring thread.
    monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
    monitor_thread.start()

    # Record start time.
    start_time = time.time()
    
    # Parallelize the training loop using joblib.
    results = Parallel(n_jobs=-1)(
        delayed(train_single_model_joblib)(
            i, train_circuits, train_classes, test_circuits, test_classes,
            estimator_qnn, maxiter, base_save_path
        ) for i in range(num_points)
    )
    
    # Compute elapsed time.
    elapsed_time = time.time() - start_time

    # Stop the CPU monitoring thread.
    stop_event.set()
    monitor_thread.join()

    # Aggregate scores.
    training_scores = [None] * num_points
    test_scores = [None] * num_points
    for run_index, train_score, test_score in results:
        training_scores[run_index] = train_score
        test_scores[run_index] = test_score

    # Save the aggregate scores.
    np.savetxt(os.path.join(base_save_path, "training_scores.txt"), np.array(training_scores))
    np.savetxt(os.path.join(base_save_path, "test_scores.txt"), np.array(test_scores))

    print(f"Time to solution: {elapsed_time:.2f} seconds")
    print(f"Maximum CPU usage recorded: {max_cpu_usage[0]:.1f}%")

