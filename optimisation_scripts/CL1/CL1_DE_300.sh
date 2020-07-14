#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./CL1_optimise.py -m DE -b 300 > outputs/TL2_DE_300.out
