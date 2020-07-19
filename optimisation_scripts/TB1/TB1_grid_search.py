import sys
import os
import argparse

from brian2 import *
from brian2tools import *

import matplotlib.pyplot as plt

import cx_rate
import trials
import plotter


from cx_spiking.constants import *

import cx_spiking.plotting
import cx_spiking.inputs
import cx_spiking.network_creation as nc

import cx_spiking.optimisation.metric as metric
import cx_spiking.optimisation.ng_optimiser as ng_optimiser

import itertools
import multiprocessing


print('****** Imports completed *******')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', required=True, type=int, default=0)

# parser.add_argument('-m', '--method', required=True, type=str, default='TwoPointsDE', help='Optimiser [DE, PSO, SQP, TwoPointsDE]')
# parser.add_argument('-b', '--budget', required=True, type=int, default=300, help='Budget for the optimiser')
args = parser.parse_args()

# method = args.method
# budget = args.budget

tauE_s_full = [0.1, 0.5, 1, 1.5, 2] # ms
wE_s_full = [550, 600, 650, 700, 750] # nS
tauI_s_full = [0.1, 0.5, 1, 1.5, 2] # ms
wI_s_full = [550, 600, 650, 700, 750] # nS

print(f'args index = {args.index}   -   tauE = {tauE_s_full[args.index]}')
# print(f'method {method} - budget {budget}')

######################################
### INPUTS
######################################
route_file = os.path.join(os.environ.get('MSC_PROJECT'), 'notebooks/data/route.npz')
T_outbound = 1500

h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True, route_file=route_file, load_route=True)

cx_spiking.inputs.save_route(route_file, h, v, save_route=True)

# Convert headings
headings = cx_spiking.inputs.compute_headings(h, N=N_TL2//2, vmin=5, vmax=100)
headings = np.tile(headings, 2)

# Convert velocity into optical flow
flow = cx_spiking.inputs.compute_flow(h, v, baseline=50, vmin=0, vmax=50)


######################################
### RATE BASED CX
######################################
noise = 0.1
cx = cx_rate.CXRatePontin(noise=noise)

h, v, cx_log, cpu4_snapshot = trials.run_trial(logging=True,
                                               T_outbound=T_outbound,
                                               T_inbound=0,
                                               noise=noise,
                                               cx=cx,
                                               route=(h[:T_outbound], v[:T_outbound]))




######################################
### SPIKE BASED CX
######################################
start_scope()

time_step = 20 # ms

h_stimulus = TimedArray(headings*Hz, dt=1.*time_step*ms)
P_HEADING = PoissonGroup(N_TL2, rates='h_stimulus(t,i)')

# Neuron group
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2_source_network')
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, CL1_neuron_params, name='CL1_source_network')
G_TB1 = nc.generate_neuron_groups(N_TB1, eqs, threshold_eqs, reset_eqs, neuron_params, name='TB1_source_network')


print(f'TL2_neuron_params {TL2_neuron_params}')
print(f'CL1_neuron_params {CL1_neuron_params}')

# Add monitors
#STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_source')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=H_TL2_synapses_params, on_pre=synapses_eqs_ex)
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, 
                                params=TL2_CL1_synapses_params, on_pre=synapses_eqs_ex)
S_CL1_TB1 = nc.connect_synapses(G_CL1, G_TB1, W_CL1_TB1, model=synapses_model, 
                                params=synapses_params, on_pre=synapses_eqs_ex)
S_TB1_TB1 = nc.connect_synapses(G_TB1, G_TB1, W_TB1_TB1, model=synapses_model, 
                                params=synapses_params, on_pre=synapses_eqs_in)

print(f'H_TL2_synapses_params {H_TL2_synapses_params}')
print(f'TL2_CL1_synapses_params {TL2_CL1_synapses_params}')


#### Target

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
TB1_stimulus = TimedArray(TB1_spike_rates*cx_log.tb1.T*Hz, dt=1.*time_step*ms)
P_TB1 = PoissonGroup(N_TB1, rates='TB1_stimulus(t,i)')
# SPM_TL2_IDEAL = SpikeMonitor(P_TL2, name='TL2_target')

store('initialised')

######################################
### OPTIMISER
######################################
def run_simulation_TB1(tauE_, wE_, tauI_, wI_, 
                       Group, Target,
                       pre_Synapses, post_Synapses,
                       time, dt_, delta, rate_correction): 
    restore('initialised') 

    # set the parameters 
    Group.set_states({'tauE' : tauE_*ms,
                      'tauI' : tauI_*ms})
    print(f'taueE: {tauE_} - tauI {tauI_}')

    pre_Synapses.set_states({'wE' : wE_*nS,
                             'wI' : wI_*nS})
    post_Synapses.set_states({'wE' : wE_*nS,
                              'wI' : wI_*nS})
    print(f'wE: {wE_} - wI {wI_}')

    _, model_spike_monitor = nc.add_monitors(Group)
    target_spike_monitor = SpikeMonitor(Target, name='TB1_target_spike_monitor')
    
    run(time)

    gf = metric.compute_gamma_factor(model_spike_monitor, target_spike_monitor, time, 
                                     dt_=dt_, delta=delta, rate_correction=rate_correction)
    
    print(f'Gamma factor: {gf}')

    return gf

gamma_factors = np.zeros((len(tauE_s_full), len(wE_s_full), len(tauI_s_full), len(wI_s_full)))

delta = 1*ms
rate_correction = True

# paramlist = list(itertools.product(tauE_s_full,wE_s_full,tauI_s_full,wI_s_full))

# def func(params):
#     gf = run_simulation_TB1(params[0], params[0], params[0], params[0], 
#                             G_TB1, P_TB1,
#                             S_CL1_TB1, S_TB1_TB1, 
#                             T_outbound*time_step*ms, 
#                             defaultclock.dt, delta, rate_correction)
#     return gf

# pool = multiprocessing.Pool(processes=5)

#Distribute the parameter sets evenly across the cores
#res  = pool.map(func, paramlist)

tauE_ = tauE_s_full[args.index]
for we_, wE_ in enumerate(wE_s_full):
    for ti_, tauI_ in enumerate(tauI_s_full):
        for wi_, wI_ in enumerate(wI_s_full):
            gamma_factors[args.index, we_, ti_, wi_] = run_simulation_TB1(tauE_, wE_, tauI_, wI_, 
                                                                    G_TB1, P_TB1,
                                                                    S_CL1_TB1, S_TB1_TB1, 
                                                                    T_outbound*time_step*ms, 
                                                                    defaultclock.dt, delta, rate_correction)

with open(f'outputs/TB1_gamma_factors_grid_search_{args.index}.npz', 'wb') as f:
    np.save(f, gamma_factors)
#np.savetxt('outputs/TB1_gamma_factors_grid_search.csv', gamma_factors, delimiter=',')

candidate = np.argwhere(gamma_factors == np.min(gamma_factors))[0]

print('Final Candidate')
print(candidate, gamma_factors[candidate[0], candidate[1], candidate[2], candidate[3]])
print(tauE_s_full[candidate[0]], wE_s_full[candidate[1]], tauI_s_full[candidate[2]], wI_s_full[candidate[3]])

# ######################################
# ### TEST
# ######################################

# start_scope()

# time_step = 20 # ms


# params_TB1 = neuron_params
# synapses_CL1_TB1 = synapses_params
# synapses_TB1_TB1 = synapses_params

# params_TB1['tauE'] = tauE_s_full[candidate[0]] * ms
# synapses_CL1_TB1['wE'] = wE_s_full[candidate[1]] * nS

# params_TB1['tauI'] = tauE_s_full[candidate[2]] * ms
# synapses_TB1_TB1['wI'] = wE_s_full[candidate[3]] * nS

# # params_CL1['tauI'] = 1 * ms
# # synapses_CL1['wI'] = 300 * nS

# print(f'params_TB1 {params_TB1}')
# print(f'synapses_CL1_TB1 {synapses_CL1_TB1}')
# print(f'synapses_TB1_TB1 {synapses_TB1_TB1}')


# h_stimulus = TimedArray(headings*Hz, dt=1.*time_step*ms)
# P_HEADING = PoissonGroup(N_TL2, rates='h_stimulus(t,i)')


# # Neuron group
# G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2_test')
# G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, CL1_neuron_params, name='CL1_test')
# G_TB1 = nc.generate_neuron_groups(N_TB1, eqs, threshold_eqs, reset_eqs, params_TB1, name='TB1_test')

# # Add monitors
# STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_test')
# STM_CL1, SPM_CL1 = nc.add_monitors(G_CL1, name='CL1_test')
# STM_TB1, SPM_TB1 = nc.add_monitors(G_TB1, name='TB1_test')


# # Connect heading to TL2
# S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
#                                       params=H_TL2_synapses_params, on_pre=synapses_eqs_ex)
# S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, 
#                                 params=TL2_CL1_synapses_params, on_pre=synapses_eqs_ex)                            
# S_CL1_TB1 = nc.connect_synapses(G_CL1, G_TB1, W_CL1_TB1, model=synapses_model, 
#                                 params=synapses_CL1_TB1, on_pre=synapses_eqs_ex)
# S_TB1_TB1 = nc.connect_synapses(G_TB1, G_TB1, W_TB1_TB1, model=synapses_model, 
#                                 params=synapses_TB1_TB1, on_pre=synapses_eqs_in)


# # Run simulation
# run(T_outbound*time_step*ms)

# cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tl2, TL2_spike_rates, SPM_TL2, 
#                                             time_step, figsize=(13,8), savefig_='plots/TL2_grid_search.pdf')
# cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.cl1, CL1_spike_rates, SPM_CL1, 
#                                             time_step, figsize=(13,8), savefig_=f'plots/CL1_grid_search.pdf')
# cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tb1, TB1_spike_rates, SPM_TB1, 
#                                             time_step, figsize=(13,8), savefig_=f'plots/TB1_grid_search.pdf')
# #cx_spiking.plotting.plot_gamma_factors(gamma_factors, tauI_s_full, wI_s_full, 
# #                                       title='TB1', xlabel='wI (nS)', ylabel='tauI (ms)', 
# #                                       figsize=(11,7), savefig_='plots/TB1_gamma_factors_grid_search.pdf')
                               

print('***********  DONE  ***********')