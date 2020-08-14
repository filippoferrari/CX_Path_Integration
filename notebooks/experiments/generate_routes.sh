#!/bin/bash
source ~/.bashrc
conda activate py_msc
nice -n 12 python ./generate_routes.py > generate_routes.out
