#!/bin/bash
#SBATCH --job-name=TUPS               # Job name
#SBATCH --error=error-%j.txt             # Standard error file
#SBATCH --output=output-%j.txt           # Standard output file
#SBATCH --partition=SKLLong     # Partition or queue name
#SBATCH --time=120:00:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH -n 32                          # Number of tasks

RUNFILE=../qml_auto_test.py
export split=1
export ansatz_layers=1
export dataset='hydrocarbon_oxygen_reordered_master'
export maxiter=2000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='crx'
export ansatz_entanglement='pairwise'
export num_points=100
export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=32

source /lustre/scafellpike/local/HT06336/exa01/cxb47-exa01/qc1/bin/activate
python3 $RUNFILE