#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TN2_grid_search.py > outputs_new/TN2_grid_search.out
