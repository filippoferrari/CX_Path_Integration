#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TB1_grid_search.py -i 3 > outputs/TB1_grid_search_3.out
