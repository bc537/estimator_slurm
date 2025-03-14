#!/bin/bash

RUNFILE=../qml_auto.py
export split=4
export ansatz_layers=1
export dataset='hydrocarbon_series'
export maxiter=1000
export operator='ZZZZZZZZZZ'
export two_local_initial_layer='ry'
export two_local_entangling_layer='cz'
export ansatz_entanglement='linear'
export num_points=5

python3 $RUNFILE 