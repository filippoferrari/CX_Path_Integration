import sys
import os
import argparse 

from brian2 import *
from brian2tools import *

import matplotlib.pyplot as plt
import scipy

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
parser.add_argument('-p', '--live_plot', default=False, action="store_true")
args = parser.parse_args()

live_plot = args.live_plot

print(f'Live plot: {live_plot}')


######################################
### INPUTS
######################################
route_file = os.path.join(os.environ.get('MSC_PROJECT'), 'notebooks/data/route.npz')
T_outbound = 1500
T_inbound = 1500

h, v, = cx_spiking.inputs.generate_route(T_outbound=1500, vary_speed=True, route_file=route_file, load_route=True)

cx_spiking.inputs.save_route(route_file, h, v, save_route=True)

# Convert headings
headings = cx_spiking.inputs.compute_headings(h, N=N_TL2//2, vmin=5, vmax=100)
headings = np.tile(headings, 2)
headings = np.concatenate((headings, np.zeros(headings.shape)), axis=0)


# Convert velocity into optical flow
flow = cx_spiking.inputs.compute_flow(h, v, baseline=50, vmin=0, vmax=50)
flow = np.concatenate((flow, np.zeros(flow.shape)), axis=0)

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

time_step = 20 # ms

h_stimulus = TimedArray(headings*Hz, dt=1.*time_step*ms)
P_HEADING = PoissonGroup(N_TL2, rates='h_stimulus(t,i)')
SPM_H = SpikeMonitor(P_HEADING)

f_stimulus = TimedArray(flow*Hz, dt=1.*time_step*ms)
P_FLOW = PoissonGroup(N_TN2, rates='f_stimulus(t,i)')
SPM_FLOW = SpikeMonitor(P_FLOW)


# Neuron groups already optimised
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2_source_network')
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, CL1_neuron_params, name='CL1_source_network')
G_TB1 = nc.generate_neuron_groups(N_TB1, eqs, threshold_eqs, reset_eqs, TB1_neuron_params, name='TB1_source_network')
G_TN2 = nc.generate_neuron_groups(N_TN2, eqs, threshold_eqs, reset_eqs, TN2_neuron_params, name='TN2_source_network')

# Synapses optimised
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, 
                                      params=H_TL2_synapses_params, on_pre=synapses_eqs_ex)
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, 
                                params=TL2_CL1_synapses_params, on_pre=synapses_eqs_ex)
S_CL1_TB1 = nc.connect_synapses(G_CL1, G_TB1, W_CL1_TB1, model=synapses_model, 
                                params=CL1_TB1_synapses_params, on_pre=synapses_eqs_ex)
S_TB1_TB1 = nc.connect_synapses(G_TB1, G_TB1, W_TB1_TB1, model=synapses_model, 
                                params=TB1_TB1_synapses_params, on_pre=synapses_eqs_in)
S_P_FLOW_TN2 = nc.connect_synapses(P_FLOW, G_TN2, W_FLOW_TN2, model=synapses_model, 
                                   params=F_TN2_synapses_params, on_pre=synapses_eqs_ex)

SPM_TB1 = SpikeMonitor(G_TB1)
SPM_TN2 = SpikeMonitor(G_TN2)


######################################
### NETWORK OPERATIONS
######################################
global ref_angles, heading_angles, velocities 

ref_angles = np.linspace(-np.pi+np.pi/8, np.pi+np.pi/8, N_TB1, endpoint=False)
max_velocity = 12 

heading_angles = np.zeros(T_outbound)
velocities = np.zeros((T_outbound, 2))



def extract_spike_counts(SPM, t, time_step):
    spike_trains = SPM.spike_trains()
    neurons = np.zeros(len(SPM.spike_trains()), dtype=int)
    for idx in range(len(spike_trains)):
        spike_train = spike_trains[idx]
        neurons[idx] = len(spike_train[(spike_train > t-time_step*ms) & (spike_train < t)])
    return neurons


def circular_weighted_mean(weights, angles):
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight
    mean = math.degrees(math.atan2(y, x))
    return mean


def make_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@network_operation(dt=time_step*ms)
def extract_heading(t):
    global ref_angles, heading_angles

    timestep = int((t/ms + 0.5) / time_step)
    
    if t < time_step*ms:
        neurons = [0,0,0,1,1,0,0,0]
        tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons[i])]
        heading_angles[timestep] = make_angle(scipy.stats.circmean(tmp, low=-np.pi, high=np.pi))
        return
    neurons = extract_spike_counts(SPM_TB1, t, time_step)    
    
    if np.sum(neurons) > 0:
        # trick to get the correct weighted average of where the heading is
        # create a list with all the angles between [-pi,pi] repeated by their count
        # so [0,2,0,0,1,0,0,1] will be [-1.963, -1.963, 0.392, 2.748] and then compute
        # circular mean between [-pi, pi]
        tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons[i])]
        # -pi/8 because we center the neurons at the center of their pi/4 receptive fields
        heading_angles[timestep] = make_angle(scipy.stats.circmean(tmp, low=-np.pi, high=np.pi) - np.pi/8)
    else:
        heading_angles[timestep] = heading_angles[timestep-1]


@network_operation(dt=time_step*ms)
def extract_velocity(t):
    global velocities, max_velocity
    
    timestep = int((t/ms + 0.5) / time_step)
    if t < time_step*ms:
        velocities[timestep] = [0,0]
        return
    neurons_responses = extract_spike_counts(SPM_TN2, t, time_step)
    neurons_responses = np.clip(neurons_responses, 0, max_velocity)
    velocities[timestep] = neurons_responses / max_velocity


######################################
### PLOTTING
######################################
map_size = 800

global bee_plot, bee_x, bee_y, bee_coords

bee_coords = np.zeros((T_outbound,2))
bee_x = 0
bee_y = 0


@network_operation(dt=time_step*ms)
def update_bee_position(t):
    global bee_x, bee_y, bee_coords, heading_angles, velocities
    
    if t < time_step*ms:
        return

    # 0.5 added for numerical stability, python integer rounds down
    # sometimes t/ms = x.99999 and it gets rounded to x-1
    timestep = int((t/ms + 0.5) / time_step)

    # Compute speed component
    speed = 1 + np.clip(np.linalg.norm(velocities[timestep]), 0, 1) 

    angle = heading_angles[timestep]
    # x should be cos and y should be sin 
    # keep compatibility with stone's code (plotter.py : line 79)
    x_comp = np.sin(angle) * speed
    y_comp = np.cos(angle) * speed

    bee_x = bee_coords[timestep-1,0] + x_comp
    bee_y = bee_coords[timestep-1,1] + y_comp
    bee_coords[timestep,0] = bee_x 
    bee_coords[timestep,1] = bee_y 


if live_plot:
    f = plt.figure(1)
    plt.axis([-map_size,map_size,-map_size,map_size])

    #bee_plot = plt.plot(bee_coords[0,0], bee_coords[0,1], 'ko') # Vehicle

    plt.text(0, 0, 'N', fontsize=12, fontweight='heavy', color='k', ha='center', va='center')

    @network_operation(dt=time_step*ms)
    def plot_bee(t):
        global bee_plot, bee_coords, heading_angles, velocities, map_size
        
        if t < time_step*ms:
            return

        # 0.5 added for numerical stability, python integer rounds down
        # sometimes t/ms = x.99999 and it gets rounded to x-1
        timestep = int((t/ms + 0.5) / time_step)
        
        bee_plot = plt.plot([bee_coords[timestep-1,0], bee_coords[timestep,0]], 
                            [bee_coords[timestep-1,1], bee_coords[timestep,1]], 
                            'k', lw=0.5)    


        plt.axis([-map_size,map_size,-map_size,map_size])
        # plt.title(f'{timestep} - {speed} - {angle}', fontsize=9)
        plt.draw()
        plt.pause(0.01)


scheduling_summary()  

run((T_outbound)*time_step*ms, report='text')

if live_plot:
    plt.show()
else:
    f = plt.figure(1)
    plt.axis([-map_size,map_size,-map_size,map_size])

    plt.text(0, 0, 'N', fontsize=12, fontweight='heavy', color='k', ha='center', va='center')

    plt.plot(bee_coords[:,0], bee_coords[:,1], 'k', lw=0.5)
    plt.show()