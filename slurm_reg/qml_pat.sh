#!/bin/bash
# Partition to use - generally not needed
#SBATCH -p CPU
#SBATCH --time=672:0:0
#SBATCH -n 1
#SBATCH --error=error-%j.txt             # Standard error file
#SBATCH --output=output-%j.txt           # Standard output file
#SBATCH -J QC
#SBATCH --requeue 

RUNFILE=../qml_auto_reg.py
export split=1
export ansatz_layers=1
export dataset='hydrocarbon_series'
export maxiter=1000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='cz'
export ansatz_entanglement='linear'
export num_points=5
export PYTHONUNBUFFERED=TRUE

source activate qc_test
srun python3 $RUNFILE