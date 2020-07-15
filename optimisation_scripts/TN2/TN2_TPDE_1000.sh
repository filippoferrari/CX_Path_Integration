#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TN2_TPDE_1000.py > outputs/TN2_TPDE_1000.out
