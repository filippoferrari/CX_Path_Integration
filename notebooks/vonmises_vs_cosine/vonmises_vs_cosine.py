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

######################################
### SPIKING CX
######################################
mem_gain_outbound_1 = 0.05
decay_outbound_1 = 0.025
mem_gain_inbound_1 = 0.05
decay_inbound_1 = 0.033

mem_gain_outbound_2 = 0.1
decay_outbound_2 = 1.25
mem_gain_inbound_2 = 0.1
decay_inbound_2 = 1

rotation_factor = 0.1


for experiment in range(1):
    h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True)#, route_file=route_file, load_route=True)

    ######################################
    ### RATE BASED CX
    ######################################
    noise = 0.1
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
    with open(os.path.join(os.environ.get('MSC_PROJECT'), f'notebooks/vonmises_vs_cosine/experiments/route_{experiment}.pickle'), 'wb') as fh:
        pickle.dump(to_save, fh, protocol=3)

    #######################
    #### CPU4 METHOD 1
    #######################
    spiking_cx_vm = CX_SPIKING(eqs, threshold_eqs, reset_eqs,
                               h, v, 
                               mem_gain_outbound_1, decay_outbound_1, mem_gain_inbound_1, decay_inbound_1,
                               rotation_factor, time_step=20, T_outbound=T_outbound,
                               T_inbound=T_inbound, 
                               headings_method='vonmises', cpu4_method=1,
                               only_tuned_network=True)
    spiking_cx_cos = CX_SPIKING(eqs, threshold_eqs, reset_eqs,
                                h, v, 
                                mem_gain_outbound_1, decay_outbound_1, mem_gain_inbound_1, decay_inbound_1,
                                rotation_factor, time_step=20, T_outbound=T_outbound,
                                T_inbound=T_inbound, 
                                headings_method='cosine', cpu4_method=1,
                                only_tuned_network=True)

    spiking_cx_vm.run_outbound()
    spiking_cx_cos.run_outbound()
    
    to_save = {}
    to_save['spiking_cx_vm'] = spiking_cx_vm.extract_data()
    to_save['spiking_cx_cos'] = spiking_cx_cos.extract_data()
    with open(os.path.join(os.environ.get('MSC_PROJECT'), f'notebooks/vonmises_vs_cosine/experiments/exp_{experiment}_cpu4_1.pickle'), 'wb') as fh:
        pickle.dump(to_save, fh, protocol=3)

    #######################
    #### CPU4 METHOD 2
    #######################
    spiking_cx_vm = CX_SPIKING(eqs, threshold_eqs, reset_eqs,
                               h, v, 
                               mem_gain_outbound_2, decay_outbound_2, mem_gain_inbound_2, decay_inbound_2,
                               rotation_factor, time_step=20, T_outbound=T_outbound,
                               T_inbound=T_inbound, 
                               headings_method='vonmises', cpu4_method=2,
                               only_tuned_network=True)
    spiking_cx_cos = CX_SPIKING(eqs, threshold_eqs, reset_eqs,
                                h, v, 
                                mem_gain_outbound_2, decay_outbound_2, mem_gain_inbound_2, decay_inbound_2,
                                rotation_factor, time_step=20, T_outbound=T_outbound,
                                T_inbound=T_inbound, 
                                headings_method='cosine', cpu4_method=2,
                                only_tuned_network=True)

    spiking_cx_vm.run_outbound()
    spiking_cx_cos.run_outbound()

    to_save = {}
    to_save['spiking_cx_vm'] = spiking_cx_vm.extract_data()
    to_save['spiking_cx_cos'] = spiking_cx_cos.extract_data()
    with open(os.path.join(os.environ.get('MSC_PROJECT'), f'notebooks/vonmises_vs_cosine/experiments/exp_{experiment}_cpu4_2.pickle'), 'wb') as fh:
        pickle.dump(to_save, fh, protocol=3)

