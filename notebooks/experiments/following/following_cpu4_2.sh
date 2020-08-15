#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./following_cpu4_2.py > following_cpu4_2.out
