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


print('****** Imports completed *******')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', required=True, type=str, default='TwoPointsDE', help='Optimiser [DE, PSO, SQP, TwoPointsDE]')
parser.add_argument('-b', '--budget', required=True, type=int, default=300, help='Budget for the optimiser')
args = parser.parse_args()

method = args.method
budget = args.budget

print(f'method {method} - budget {budget}')

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
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, neuron_params, name='CL1_source_network')

# Add monitors
#STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_source')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=TL2_synapses_params, on_pre=synapses_eqs_ex)
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, 
                                params=synapses_params, on_pre=synapses_eqs_ex)


#### Target
TL2_spike_rates = 90 # Hz
CL1_spike_rates = 50 # Hz

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
CL1_stimulus = TimedArray(CL1_spike_rates*cx_log.cl1.T*Hz, dt=1.*time_step*ms)
P_CL1 = PoissonGroup(N_CL1, rates='CL1_stimulus(t,i)')
# SPM_TL2_IDEAL = SpikeMonitor(P_TL2, name='TL2_target')

store('initialised')

######################################
### OPTIMISER
######################################
def run_simulation_CL1(tauE_, wE_, tauI_, wI_, Group, Synapses, Target,
                       time, dt_, delta, rate_correction): 
    restore('initialised') 

    # set the parameters 
    Group.set_states({'tauE' : tauE_*ms,
                      'tauI' : tauI_*ms})
    print(f'taueE: {tauE_} - tauI {tauI_}')

    Synapses.set_states({'wE' : wE_*nS,
                         'wI' : wI_*nS})
    print(f'wE: {wE_} - wI {wI_}')

    _, model_spike_monitor = nc.add_monitors(Group)
    target_spike_monitor = SpikeMonitor(Target, name='CL1_target_spike_monitor')
    
    run(time)

    gf = metric.compute_gamma_factor(model_spike_monitor, target_spike_monitor, time, 
                                     dt_=dt_, delta=delta, rate_correction=rate_correction)
    
    print(f'Gamma factor: {gf}')

    return gf



# Values to optimise
bounds = [[0.1,5],     # tauE [ms]
          [200,1000]] # wE   [nS]
        #   [0.1,5],     # tauI [ms]
        #   [200,1000]] # wI   [nS]


# Other fixed arguments to optimisation function
delta = 1*ms
rate_correction = True

args = [
        1,                       # tauI - useless for this optimisation
        300,                     # wI - useless for this optimisation
        G_CL1,                   # neuron group to optimise
        S_TL2_CL1,               # synapses to optimise
        P_CL1,                   # target population
        T_outbound*time_step*ms, # simulation time
        defaultclock.dt,         # simulation time step
        delta,                   # time window for gamma factor
        rate_correction          # apply rate correction to gamma factor
       ]




# Set instruments
instruments = ng_optimiser.set_instrumentation(bounds, args)
optim = ng_optimiser.set_optimiser(instruments, method=method, budget=budget)

optim_min, recommendation = ng_optimiser.run_optimiser(optim, run_simulation_CL1, verbosity=2)

candidate = optim_min.provide_recommendation()

print(candidate.args)


######################################
### TEST
######################################

start_scope()

time_step = 20 # ms


params_CL1 = neuron_params
synapses_CL1 = synapses_params

params_CL1['tauE'] = candidate.args[0] * ms
synapses_CL1['wE'] = candidate.args[1] * nS

# params_CL1['tauI'] = 1 * ms
# synapses_CL1['wI'] = 300 * nS

print(params_CL1)
print(synapses_CL1)


h_stimulus = TimedArray(headings*Hz, dt=1.*time_step*ms)
P_HEADING = PoissonGroup(N_TL2, rates='h_stimulus(t,i)')


# Neuron group
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2_test')
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, neuron_params, name='CL1_test')

# Add monitors
STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_test')
STM_CL1, SPM_CL1 = nc.add_monitors(G_CL1, name='CL1_test')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=TL2_synapses_params, on_pre=synapses_eqs_ex)
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, 
                                params=synapses_params, on_pre=synapses_eqs_ex)


# Run simulation
run(T_outbound*time_step*ms)

cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tl2, TL2_spike_rates, SPM_TL2, 
                                            time_step, figsize=(13,8), savefig_=f'plots/TL2_{method}_{budget}_optimised.pdf')
cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.cl1, CL1_spike_rates, SPM_CL1, 
                                            time_step, figsize=(13,8), savefig_=f'plots/CL1_{method}_{budget}_optimised.pdf')
