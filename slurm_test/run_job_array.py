import os
import yaml

# Load configuration from YAML file
with open("config_scatter.yaml", "r") as f:
    config = yaml.safe_load(f)

split = config["split"]
ansatz_layers = config["ansatz_layers"]
num_data_points = config["num_data_points"][0]  # Assuming num_data_points contains only one value

template = """#!/bin/sh
#BSUB -q scafellpikeSKL
#BSUB -W 02:00
#BSUB -o %J-%I.out
#BSUB -e %J-%I.err
#BSUB -R "span[hosts=1]"
#BSUB -n 1
#BSUB -J "jobarray[1-{TOTAL_JOBS}]"

split=({split})
ansatz_layers=({ansatz_layers})

# Calculate the indices
INDEX=$((LSB_REMOTEINDEX - 1))
NUM_ANSATZ_LAYERS={NUM_ANSATZ_LAYERS}
NUM_SPLIT={NUM_SPLIT}
SPLIT_INDEX=$((INDEX / (NUM_ANSATZ_LAYERS * {NUM_DATA_POINTS})))
ANSATZ_LAYERS_INDEX=$(((INDEX / {NUM_DATA_POINTS}) % NUM_ANSATZ_LAYERS))
DATA_POINTS_INDEX=$((INDEX % {NUM_DATA_POINTS}))

# Assign individual values
split=${{split[SPLIT_INDEX]}}
ansatz_layers=${{ansatz_layers[ANSATZ_LAYERS_INDEX]}}

RUNFILE=/lustre/scafellpike/local/HT06336/exa01/dxm15-exa01/estimator_slurm/qml_auto.py

ENV_VARS_FILE=env_vars_${{LSB_REMOTEJID}}_${{LSB_REMOTEINDEX}}.sh

# Print the constructed name for debugging
echo "Constructed ENV_VARS_FILE: $ENV_VARS_FILE"

echo "split=$split" > $ENV_VARS_FILE
echo "ansatz_layers=$ansatz_layers" >> $ENV_VARS_FILE
echo "dataset='hydrocarbon_oxygen_reordered_master'" >> $ENV_VARS_FILE
echo "maxiter=2000" >> $ENV_VARS_FILE
echo "operator='IIIIZZIIII'" >> $ENV_VARS_FILE
echo "two_local_initial_layer='ry'" >> $ENV_VARS_FILE
echo "two_local_entangling_layer='crx'" >> $ENV_VARS_FILE
echo "ansatz_entanglement='pairwise'" >> $ENV_VARS_FILE
echo "num_points={NUM_DATA_POINTS}" >> $ENV_VARS_FILE
echo "PYTHONUNBUFFERED=TRUE" >> $ENV_VARS_FILE
echo "OMP_NUM_THREADS=1" >> $ENV_VARS_FILE
echo "LSB_REMOTEJID=$LSB_REMOTEJID" >> $ENV_VARS_FILE
echo "DATA_POINTS_INDEX=$DATA_POINTS_INDEX" >> $ENV_VARS_FILE

#source /lustre/scafellpike/local/HT06336/exa01/cxb47-exa01/qc1/bin/activate
source /lustre/scafellpike/local/HT06336/exa01/dxm15-exa01/estimator_slurm/venv-me/bin/activate

# Echo start time
echo "Start time: $(date)"

# Debug print statements before running Python script
echo "Debug: split=$split"
echo "Debug: ansatz_layers=$ansatz_layers"

# run qml_auto
python3 $RUNFILE $ENV_VARS_FILE

# Echo end time
echo "End time: $(date)"

# Debug print statements
echo "Debug: LSB_REMOTEJID=$LSB_REMOTEJID"
echo "Debug: LSB_REMOTEINDEX=$LSB_REMOTEINDEX"
echo "Debug: INDEX=$INDEX"
echo "Debug: SPLIT_INDEX=$SPLIT_INDEX"
echo "Debug: ANSATZ_LAYERS_INDEX=$ANSATZ_LAYERS_INDEX"
echo "Debug: DATA_POINTS_INDEX=$DATA_POINTS_INDEX"
echo "Debug: split=$split"
echo "Debug: ansatz_layers=$ansatz_layers"

# delete tmp var file
rm $ENV_VARS_FILE
"""

TOTAL_JOBS = len(split) * len(ansatz_layers) * num_data_points

with open('job_array.bsub', 'w') as tmp:
    tmp.write(template.format(
        TOTAL_JOBS=TOTAL_JOBS,
        split=" ".join(map(str, split)),
        ansatz_layers=" ".join(map(str, ansatz_layers)),
        NUM_ANSATZ_LAYERS=len(ansatz_layers),
        NUM_SPLIT=len(split),
        NUM_DATA_POINTS=num_data_points
    ))

os.system("bsub < job_array.bsub")
