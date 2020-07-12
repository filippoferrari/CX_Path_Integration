#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_TPDE_300.py > ./TL2_TPDE_300.out
