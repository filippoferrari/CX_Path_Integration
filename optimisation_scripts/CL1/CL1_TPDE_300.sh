#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./CL1_optimise.py -m TwoPointsDE -b 300 > outputs/CL1_TPDE_300.out
