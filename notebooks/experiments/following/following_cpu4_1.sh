#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./following_cpu4_1.py > following_cpu4_1.out
