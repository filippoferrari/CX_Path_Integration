#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_DE_300.py > outputs/TL2_DE_300.out
