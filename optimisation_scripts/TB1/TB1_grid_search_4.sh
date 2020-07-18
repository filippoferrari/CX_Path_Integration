#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TB1_grid_search.py -i 4 > outputs/TB1_grid_search_4.out
