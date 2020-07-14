#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_TPDE_1000.py > outputs/TL2_TPDE_1000.out
