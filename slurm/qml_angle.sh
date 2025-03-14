#!/bin/bash
#SBATCH --job-name=TUPS               # Job name
#SBATCH --error=error-%j.txt             # Standard error file
#SBATCH --output=output-%j.txt           # Standard output file
#SBATCH --partition=scafellpikeSKL    # Partition or queue name
#SBATCH --time=48:00:00               # Maximum runtime (D-HH:MM:SS)
#SBATCH -n 1                          # Number of tasks

RUNFILE=../qml_auto_angle.py
export split=1
export feature_layers=1
export ansatz_layers=1
export dataset='hydrocarbon_series'
export maxiter=1000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='cz'
export ansatz_entanglement='linear'
export num_points=100
export PYTHONUNBUFFERED=TRUE

source /lustre/scafellpike/local/HT06336/exa01/cxb47-exa01/qc1/bin/activate
srun python3 $RUNFILE