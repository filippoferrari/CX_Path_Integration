#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./CL1_grid_search.py > outputs/CL1_grid_search.out
