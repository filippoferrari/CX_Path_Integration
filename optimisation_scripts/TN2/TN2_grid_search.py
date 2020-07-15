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
    print(f'Candidate(args=({tauE_}, {wE_}')

    return gf


tauE_s = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # ms
wE_s = [200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 950, 1000] # nS

gamma_factors = np.zeros((len(tauE_s), len(wE_s)))

delta = 1*ms
rate_correction = True


for t, tauE in enumerate(tauE_s):
    for w, wE in enumerate(wE_s):
        gamma_factors[t,w] = run_simulation_TN2(tauE, wE, 1, 300, G_TN2,
                                               S_P_FLOW_TN2, P_TN2, T_outbound*time_step*ms, 
                                               defaultclock.dt, delta, rate_correction)


np.savetxt('outputs/TN2_gamma_factors_grid_search.csv', gamma_factors, delimiter=',')

candidate = np.argwhere(gamma_factors == np.min(gamma_factors))[0]

print('Final Candidate')
print(candidate, gamma_factors[candidate[0]], gamma_factors[candidate[1]])

######################################
### TEST
######################################

start_scope()

time_step = 20 # ms

f_stimulus = TimedArray(flow*Hz, dt=1.*time_step*ms)
P_FLOW = PoissonGroup(N_TN2, rates='f_stimulus(t,i)')


params_TL2 = neuron_params
synapses_TL2 = synapses_params

params_TL2['tauE'] = tauE_s[candidate[0]] * ms
synapses_TL2['wE'] = wE_s[candidate[1]] * nS

# params_TL2['tauI'] = 1 * ms
# synapses_TL2['wI'] = 300 * nS


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

cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tn2, TN2_spike_rates, SPM_TN2, 
                                            time_step, figsize=(13,8), savefig_='plots/TN2_grid_search.pdf')
cx_spiking.plotting.plot_gamma_factors(gamma_factors, tauE_s, wE_s, 
                                       figsize=(11,7), savefig_='plots/TN2_gamma_factors_grid_search.pdf'):
