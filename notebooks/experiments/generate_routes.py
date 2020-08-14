import sys
import os
import pickle

from brian2 import *
from brian2tools import *

import matplotlib.pyplot as plt
import scipy

import cx_rate
import trials
import plotter
import bee_simulator

from cx_spiking.central_complex import CX_SPIKING
from cx_spiking.constants import *
import cx_spiking 

######################################
### INPUTS
######################################
route_file = os.path.join(os.environ.get('MSC_PROJECT'), 'notebooks/data/route.npz')
T_outbound = 1500
T_inbound = 1500
T = T_outbound+T_inbound
noise = 0.1

numpy.random.seed(1556895)

for experiment in range(100):
    h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True)#, route_file=route_file, load_route=True)

    ######################################
    ### RATE BASED CX
    ######################################
    cx = cx_rate.CXRatePontin(noise=noise)

    h, v, cx_log, cpu4_snapshot = trials.run_trial(logging=True,
                                                   T_outbound=T_outbound,
                                                   T_inbound=T_inbound,
                                                   noise=noise,
                                                   cx=cx,
                                                   route=(h[:T_outbound], v[:T_outbound]))

    to_save = {}
    to_save['h'] = h
    to_save['v'] = v
    to_save['cx_log'] = cx_log
    with open(os.path.join(os.environ.get('MSC_PROJECT'), f'notebooks/experiments/routes/route_{experiment}.pickle'), 'wb') as fh:
        pickle.dump(to_save, fh, protocol=3)
