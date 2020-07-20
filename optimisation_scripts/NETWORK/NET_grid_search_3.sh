#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./NET_grid_search.py -i 3 > outputs/NET_grid_search_3.out
