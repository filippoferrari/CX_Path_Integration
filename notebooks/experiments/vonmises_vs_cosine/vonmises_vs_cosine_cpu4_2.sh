#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./vonmises_vs_cosine_cpu4_2.py > vonmises_vs_cosine_cpu4_2.out
