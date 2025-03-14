#!/bin/bash

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

python3 $RUNFILE 