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
P_CL1 = PoissonGroup(N_CL1, rates=CL1_spike_rates*Hz)


# Neuron group
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2_source_network')
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, CL1_neuron_params, name='CL1_source_network')

# Add monitors
#STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_source')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=H_TL2_synapses_params, on_pre=synapses_eqs_ex)
S_P_CL1_CL1 = nc.connect_synapses(P_CL1, G_CL1, np.eye(N_CL1), model=synapses_model, 
                                      params=P_CL1_CL1_synapses_params, on_pre=synapses_eqs_ex) 
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, np.eye(N_CL1), model=synapses_model, 
                                params=synapses_params, on_pre=synapses_eqs_in)


#### Target

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
CL1_stimulus = TimedArray(CL1_spike_rates*cx_log.cl1.T*Hz, dt=1.*time_step*ms)
P_CL1_TARGET = PoissonGroup(N_CL1, rates='CL1_stimulus(t,i)')
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


delta = 1*ms
rate_correction = True

tauI_s = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # ms
wI_s = [200, 250, 300, 350, 400, 500, 600, 700, 800, 850, 900, 950, 1000] # nS

gamma_factors = np.zeros((len(tauE_s), len(wE_s)))

for t_, tauI_ in enumerate(tauI_s):
    for w_, wI_ in enumerate(wI_s):               #TODO HERE WITH RESULTS on tauE_ and wE_
        gamma_factors[t_,w_] = run_simulation_CL1(1, 600, tauI_, wI_, 
                                                  G_CL1,
                                                  S_TL2_CL1, 
                                                  P_CL1_TARGET, 
                                                  T_outbound*time_step*ms, 
                                                  defaultclock.dt, delta, rate_correction)

np.savetxt('outputs/CL1_gamma_factors_grid_search.csv', gamma_factors, delimiter=',')

candidate = np.argwhere(gamma_factors == np.min(gamma_factors))[0]

print('Final Candidate')
print(candidate, gamma_factors[candidate[0]], gamma_factors[candidate[1]])
