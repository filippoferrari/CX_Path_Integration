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

# Neuron group
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, neuron_params, name='TL2')

# Add monitors
STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=synapses_params, on_pre=synapses_eqs_ex)


#### Target
TL2_spike_rates = 90 # Hz

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
TL2_stimulus = TimedArray(TL2_spike_rates*cx_log.tl2.T*Hz, dt=1.*time_step*ms)
P_TL2 = PoissonGroup(N_TL2, rates='TL2_stimulus(t,i)')
SPM_TL2_IDEAL = SpikeMonitor(P_TL2, name='TL2_target')

store()


######################################
### OPTIMISER
######################################
# Values to optimise
bounds = [[0.1,5],     # tauE [ms]
          [200,1000]]  # wE   [nS]
        #   [0.1,5],    # tauI [ms]
        #   [200,1000]] # wI   [nS]


# Other fixed arguments to optimisation function
delta = 1*ms
rate_correction = True

args = [
        1,                       # tauI - useless for this optimisation
        300,                     # wI - useless for this optimisation
        G_TL2,                   # neuron group to optimise
        S_P_HEADING_TL2,         # synapses to optimise
        P_TL2,           # target spike monitor
        T_outbound*time_step*ms, # simulation time
        defaultclock.dt,         # simulation time step
        delta,                   # time window for gamma factor
        rate_correction          # apply rate correction to gamma factor
       ]


simulation = ng_optimiser.run_simulation_TL2

# Set instruments
instruments = ng_optimiser.set_instrumentation(bounds, args)
optim = ng_optimiser.set_optimiser(instruments, method='SQP', budget=300)

optim_min, recommendation = ng_optimiser.run_optimiser(optim, simulation, verbosity=2)

candidate = optim_min.provide_recommendation()

print(candidate.args)


######################################
### TEST
######################################

start_scope()

time_step = 20 # ms

h_stimulus = TimedArray(headings*Hz, dt=1.*time_step*ms)
P_HEADING = PoissonGroup(N_TL2, rates='h_stimulus(t,i)')


params_TL2 = neuron_params
synapses_TL2 = synapses_params

params_TL2['tauE'] = candidate.args[0] * ms
synapses_TL2['wE'] = candidate.args[1] * nS

# params_TL2['tauI'] = 1 * ms
# synapses_TL2['wI'] = 300 * nS


print(params_TL2)
print(synapses_TL2)


# Neuron group
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, params_TL2, name='TL2_test')

# Add monitors
STM_TL2, SPM_TL2 = nc.add_monitors(G_TL2, name='TL2_test')

# Connect heading to TL2
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, params=synapses_TL2, model=synapses_model, on_pre=synapses_eqs_ex)


# Run simulation
run(T_outbound*time_step*ms)

cx_spiking.plotting.plot_rate_cx_log_spikes(cx_log.tl2, TL2_spike_rates, SPM_TL2, 
                                            time_step, figsize=(13,8), savefig_='TL2_SQP_optimised.pdf')