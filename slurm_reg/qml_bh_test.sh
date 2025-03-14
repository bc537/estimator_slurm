#!/bin/bash

RUNFILE=../qml_auto_reg_bh.py
export split=1
export ansatz_layers=5
export dataset='hydrocarbon_series'
export maxiter=5000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='crx'
export ansatz_entanglement='pairwise'
export num_points=2
export PYTHONUNBUFFERED=TRUE
export bh_iterations=5
export bh_stepsize=0.001
export bh_temp=0.0

python3 $RUNFILE 