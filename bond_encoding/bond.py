import numpy as np
from rdkit import Chem
from qiskit import QuantumCircuit
import os
import json

def coulomb_matrix(smiles: str, add_hydrogens: bool = False, bond_coupling: float = 1.0, atom_factor: float = 2.4) -> np.ndarray:
    """
    Computes the adjacent Coulomb matrix for a given molecule specified by a SMILES string,
    using specific average bond lengths for adjacent atom pairs.
    
    Parameters:
    - smiles (str): The SMILES string representing the molecule.
    - add_hydrogens (bool): Whether to add hydrogen atoms to the molecule.
    
    Returns:
    - np.ndarray: The Coulomb matrix of the molecule.
    """
    # Load the molecule from the SMILES string
    molecule = Chem.MolFromSmiles(smiles)
    
    # Add hydrogen atoms if specified
    if add_hydrogens == True:
        molecule = Chem.AddHs(molecule)
    
    # Get the atomic numbers of the atoms
    atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    
    # Number of atoms
    num_atoms = len(atomic_numbers)
    
    # Initialize the Coulomb matrix
    coulomb_matrix = np.zeros((num_atoms, num_atoms))
    
    # Fill in the Coulomb matrix
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                # Diagonal elements: 0.5 * Z_i^2.4
                coulomb_matrix[i, j] = (0.5 * atomic_numbers[i] ** atom_factor) 
            else:
                # Find the bond between atoms i and j
                bond = molecule.GetBondBetweenAtoms(i, j)
                if bond:
                    bond_type = bond.GetBondType()
                    if bond_type == Chem.rdchem.BondType.SINGLE:
                        distance = 1
                    elif bond_type == Chem.rdchem.BondType.DOUBLE:
                        distance = 2
                    elif bond_type == Chem.rdchem.BondType.TRIPLE:
                        distance = 3
                    elif bond_type == Chem.rdchem.BondType.AROMATIC:
                        distance = 1.5
                    coulomb_matrix[i, j] = (atomic_numbers[i] * atomic_numbers[j] / distance) * bond_coupling 
    
    return coulomb_matrix

def matrix_to_circuit(matrix, num_qubits, n_layers: int = 1, reverse_bits: bool = False, initial_layer: str = 'rx', entangling_layer: str = 'rzz', n_atom_to_qubit: int = 1, interleaved: str = None) -> QuantumCircuit:
    """
    Converts a matrix to a QuantumCircuit object.
    
    Parameters:
    - matrix (np.ndarray): The matrix to convert.
    
    Returns:
    - QuantumCircuit: The QuantumCircuit object representing the matrix.
    """
    # Get the number of qubits required to represent the matrix
    matrix_size = matrix.shape[0]

    # Toggle reverse bits
    if reverse_bits == True:
        m = np.flip(np.arange(num_qubits - matrix_size * n_atom_to_qubit, num_qubits))
    else:
        m = np.arange(0, matrix_size * n_atom_to_qubit)
    
    m = np.reshape(m, (matrix_size, n_atom_to_qubit))

    # Initialize the QuantumCircuit object
    qc = QuantumCircuit(num_qubits)

    for _ in range(n_layers):
        for i in range(matrix_size):
            if initial_layer == 'ry':
                for k in range(n_atom_to_qubit):
                    qc.ry(matrix[i, i], m[i, k])
            elif initial_layer == 'rz':
                for k in range(n_atom_to_qubit):
                    qc.rz(matrix[i, i], m[i, k])
            else:
                for k in range(n_atom_to_qubit):
                    qc.rx(matrix[i, i], m[i, k])
        if interleaved == 'cnot' or interleaved == 'cx':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.cx(a[j], a[j + 1])
        elif interleaved == 'cz':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.cz(a[j], a[j + 1])
        elif interleaved == 'rxx':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.rxx(matrix[i, i], a[j], a[j + 1])
        elif interleaved == 'ryy':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.ryy(matrix[i, i], a[j], a[j + 1])
        elif interleaved == 'rzz':
            for i in range(matrix_size):
                a = m[i, :]
                for j in range(len(a) - 1):
                    qc.rzz(matrix[i, i], a[j], a[j + 1])
        for i in range(matrix_size):
            for j in range(matrix_size):
                if (i < j) and (matrix[i, j] != 0.0):
                    if n_atom_to_qubit == 1:
                        q_c = m[i]
                        q_t = m[j]
                        if entangling_layer == 'rxx':
                            qc.rxx(matrix[i, j], q_c, q_t)
                        elif entangling_layer == 'ryy':
                            qc.ryy(matrix[i, j], q_c, q_t)
                        else:
                            qc.rzz(matrix[i, j], q_c, q_t)
                    else:
                        q_c = m[i, -1]
                        q_t = m[j, 0]
                        if entangling_layer == 'rxx':
                            qc.rxx(matrix[i, j], q_c, q_t)
                        elif entangling_layer == 'ryy':
                            qc.ryy(matrix[i, j], q_c, q_t)
                        else:
                            qc.rzz(matrix[i, j], q_c, q_t)
    
    return qc

def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)