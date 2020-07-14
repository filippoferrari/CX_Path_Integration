#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./CL1_optimise.py -m TwoPointsDE -b 1000 > outputs/CL1_TPDE_1000.out
