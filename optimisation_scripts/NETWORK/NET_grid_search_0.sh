#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./NET_grid_search.py -i 0 > outputs/NET_grid_search_0.out
