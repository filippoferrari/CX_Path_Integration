import sys
import os

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


######################################
### INPUTS
######################################
route_file = os.path.join(os.environ.get('MSC_PROJECT'), 'notebooks/data/route.npz')
T_outbound = 1500

h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True, route_file=route_file, load_route=True)

cx_spiking.inputs.save_route(route_file, h, v, save_route=True)

# Convert headings
headings = cx_spiking.inputs.compute_headings(h, N=N_TN2//2, vmin=5, vmax=100)
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

f_stimulus = TimedArray(flow*Hz, dt=1.*time_step*ms)
P_FLOW = PoissonGroup(N_TN2, rates='f_stimulus(t,i)')

# Neuron group
G_TN2 = nc.generate_neuron_groups(N_TN2, eqs, threshold_eqs, reset_eqs, neuron_params, name='TN2_source_network')

# Add monitors
# STM_TN2, SPM_TN2 = nc.add_monitors(G_TN2, name='TN2_source')

# Connect heading to TN2
S_P_FLOW_TN2 = nc.connect_synapses(P_FLOW, G_TN2, W_FLOW_TN2, model=synapses_model, 
                                   params=synapses_params, on_pre=synapses_eqs_ex)

#### Target
TN2_spike_rates_min = 50 # Hz
TN2_spike_rates_max = 160 # Hz

rescaled_tn2 = cx_log.tn2.T * (TN2_spike_rates_max - TN2_spike_rates_min) + TN2_spike_rates_min

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
TN2_stimulus = TimedArray(rescaled_tn2*Hz, dt=1.*time_step*ms)
P_TN2 = PoissonGroup(N_TN2, rates='TN2_stimulus(t,i)')


store('initialised')


######################################
### OPTIMISER
######################################
def run_simulation_TN2(tauE_, wE_, tauI_, wI_, Group, Synapses, Target,
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
    target_spike_monitor = SpikeMonitor(Target, name='TN2_target_spike_monitor')
    
    run(time)

    gf = metric.compute_gamma_factor(model_spike_monitor, target_spike_monitor, time, 
                                     dt_=dt_, delta=delta, rate_correction=rate_correction)
    
    print(f'Gamma factor: {gf}')

    return gf


# Values to optimise
bounds = [[0.1,5],     # tauE [ms]
          [200,1000]]  # wE   [nS]
        #   [0.1,5],     # tauI [ms]
        #   [200,1000]] # wI   [nS]


# Other fixed arguments to optimisation function
delta = 1*ms
rate_correction = True

args = [
        1,                       # tauI - useless for this optimisation
        300,                     # wI - useless for this optimisation
        G_TN2,                   # neuron group to optimise
        S_P_FLOW_TN2,         # synapses to optimise
        P_TN2,                   # target population
        T_outbound*time_step*ms, # simulation time
        defaultclock.dt,         # simulation time step
        delta,                   # time window for gamma factor
        rate_correction          # apply rate correction to gamma factor
       ]




# Set instruments
instruments = ng_optimiser.set_instrumentation(bounds, args)
optim = ng_optimiser.set_optimiser(instruments, method='DE', budget=300)

optim_min, recommendation = ng_optimiser.run_optimiser(optim, run_simulation_TN2, verbosity=2)

candidate = optim_min.provide_recommendation()

print(candidate.args)


######################################
### TEST
######################################

start_scope()

time_step = 20 # ms

f_stimulus = TimedArray(flow*Hz, dt=1.*time_step*ms)
P_FLOW = PoissonGroup(N_TN2, rates='f_stimulus(t,i)')


params_TN2 = neuron_params
synapses_TN2 = synapses_params

params_TN2['tauE'] = candidate.args[0] * ms
synapses_TN2['wE'] = candidate.args[1] * nS

# params_TN2['tauI'] = 1 * ms
# synapses_TN2['wI'] = 300 * nS


print(params_TN2)
print(synapses_TN2)


# Neuron group
G_TN2 = nc.generate_neuron_groups(N_TN2, eqs, threshold_eqs, reset_eqs, params_TN2, name='TN2_test')

# Add monitors
STM_TN2, SPM_TN2 = nc.add_monitors(G_TN2, name='TN2_test')

# Connect heading to TN2
S_P_FLOW_TN2 = nc.connect_synapses(P_FLOW, G_TN2, W_FLOW_TN2, params=synapses_TN2, 
                                   model=synapses_model, on_pre=synapses_eqs_ex)


# Run simulation
run(T_outbound*time_step*ms)

cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tn2, TN2_spike_rates_max, SPM_TN2, 
                                            time_step, figsize=(13,8), savefig_='plots/TN2_DE_optimised.pdf')
