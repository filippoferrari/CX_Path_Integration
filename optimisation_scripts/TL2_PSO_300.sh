#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_PSO_300.py > ./TL2_PSO_300.out
