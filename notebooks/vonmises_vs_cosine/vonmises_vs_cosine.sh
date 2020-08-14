#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./vonmises_vs_cosine.py > vonmises_vs_cosine.out
