#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TN2_PSO_300.py > outputs/TN2_PSO_300.out
