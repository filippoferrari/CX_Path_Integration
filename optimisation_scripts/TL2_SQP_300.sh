#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_SQP_300.py > outputs/TL2_SQP_300.out
