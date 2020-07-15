#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TN2_SQP_300.py > outputs/TN2_SQP_300.out
