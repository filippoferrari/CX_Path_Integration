#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./vonmises_vs_cosine_cpu4_1.py > vonmises_vs_cosine_cpu4_1.out
