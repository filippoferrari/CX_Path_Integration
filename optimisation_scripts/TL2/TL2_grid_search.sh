#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./TL2_grid_search.py > outputs/TL2_grid_search.out
