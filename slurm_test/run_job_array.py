import os
import time
import yaml

# Load configuration from YAML file
with open("config_scatter.yaml", "r") as f:
    config = yaml.safe_load(f)

split = config["split"]
ansatz_layers = config["ansatz_layers"]

template = """#!/bin/sh
#BSUB -q scafellpikeSKL
#BSUB -W 48:00
#BSUB -o %J-%I.out
#BSUB -e %J-%I.err
#BSUB -R "span[hosts=1]"
#BSUB -n 32
#BSUB -x
#BSUB -J "jobarray[1-{TOTAL_JOBS}]"

split=({split})
ansatz_layers=({ansatz_layers})

# Calculate the indices
INDEX=$((LSB_REMOTEINDEX - 1))
NUM_ANSATZ_LAYERS={NUM_ANSATZ_LAYERS}
SPLIT_INDEX=$((INDEX / NUM_ANSATZ_LAYERS))
ANSATZ_LAYERS_INDEX=$((INDEX % NUM_ANSATZ_LAYERS))

# Assign individual values
split=${{split[SPLIT_INDEX]}}
ansatz_layers=${{ansatz_layers[ANSATZ_LAYERS_INDEX]}}

RUNFILE=../qml_auto_test.py
export split=$split
export ansatz_layers=$ansatz_layers
export dataset='hydrocarbon_oxygen_reordered_master'
export maxiter=2000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='crx'
export ansatz_entanglement='pairwise'
export num_points=100
export PYTHONUNBUFFERED=TRUE
export OMP_NUM_THREADS=32

#source /lustre/scafellpike/local/HT06336/exa01/cxb47-exa01/qc1/bin/activate
source /lustre/scafellpike/local/HT06336/exa01/dxm15-exa01/estimator_slurm/venv-me/bin/activate

# Echo start time
echo "Start time: $(date)"

python3 $RUNFILE

# Echo end time
echo "End time: $(date)"

# Debug print statements
echo "Debug: LSB_REMOTEINDEX=$LSB_REMOTEINDEX"
echo "Debug: INDEX=$INDEX"
echo "Debug: SPLIT_INDEX=$SPLIT_INDEX"
echo "Debug: ANSATZ_LAYERS_INDEX=$ANSATZ_LAYERS_INDEX"
echo "Debug: split=$split"
echo "Debug: ansatz_layers=$ansatz_layers"
"""

TOTAL_JOBS = len(split) * len(ansatz_layers)

with open('job_array.bsub', 'w') as tmp:
    tmp.write(template.format(
        TOTAL_JOBS=TOTAL_JOBS,
        split=" ".join(map(str, split)),
        ansatz_layers=" ".join(map(str, ansatz_layers)),
        NUM_ANSATZ_LAYERS=len(ansatz_layers)
    ))

os.system("bsub < job_array.bsub")
