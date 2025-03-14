#!/bin/bash

RUNFILE=../qml_auto_reg.py
export split=1
export ansatz_layers=5
export dataset='hydrocarbon_series'
export maxiter=5000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='crx'
export ansatz_entanglement='pairwise'
export num_points=5

python3 $RUNFILE 