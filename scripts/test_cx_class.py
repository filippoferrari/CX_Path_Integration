import cx_spiking.central_complex
import os
import numpy as np

from cx_spiking.central_complex import CX_SPIKING
from cx_spiking.constants import *

######################################
### INPUTS
######################################
route_file = os.path.join(os.environ.get('MSC_PROJECT'), 'notebooks/data/route.npz')
T_outbound = 1500
T_inbound = 1500
T = T_outbound+T_inbound

h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True, route_file=route_file, load_route=True)

cx_spiking.inputs.save_route(route_file, h, v, save_route=False)

# Convert headings
headings = cx_spiking.inputs.compute_headings(h, N=N_TL2//2, vmin=5, vmax=100)
headings = np.tile(headings, 2)
headings = np.concatenate((headings, np.zeros((T_inbound, headings.shape[1]))), axis=0)

# Convert velocity into optical flow
flow = cx_spiking.inputs.compute_flow(h, v, baseline=50, vmin=0, vmax=50)
flow = np.concatenate((flow, np.zeros((T_inbound, flow.shape[1]))), axis=0)


######################################
### PARAMETERS
######################################
cpu4_method = 1
mem_gain_outbound = 0.05
decay_outbound= 0.025
mem_gain_inbound = 0.05
decay_inbound = 0.033
rotation_factor = 0.1

cx = CX_SPIKING(eqs, threshold_eqs, reset_eqs,
                headings, flow, cpu4_method, 
                mem_gain_outbound, decay_outbound, mem_gain_inbound, decay_inbound,
                rotation_factor, time_step=20, T_outbound=T_outbound,
                T_inbound=T_inbound)

cx.run_outbound()