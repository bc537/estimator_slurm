from itertools import product
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
#BSUB -o output=output-%j.txt
#BSUB -e error-%j.txt
#BSUB -R "span[hosts=1]"
#BSUB -n 32
#BSUB -x
#BSUB -J "jobarray[1-{TOTAL_JOBS}]"

RUNFILE=../qml_auto_test.py
export split=({split})
export ansatz_layers=({ansatz_layers})
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
"""

TOTAL_JOBS = len(split) * len(ansatz_layers)

with open('pub_runner_array.bsub', 'w') as tmp:
    tmp.write(template.format(
    TOTAL_JOBS=TOTAL_JOBS,
    split=" ".join(map(str, split)),
    ansatz_layers=" ".join(map(str, ansatz_layers))
    ))

os.system("bsub < pub_runner_array.bsub")