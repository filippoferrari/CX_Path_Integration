#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TB1_grid_search.py -i 1 > outputs/TB1_grid_search_1.out
