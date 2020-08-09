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
args = parser.parse_args()


# tauE_s_full = [0.1, 0.5, 1, 1.5, 2] # ms
# wE_s_full = [550, 600, 650, 700, 750] # nS
# tauI_s_full = [0.1, 0.5, 1, 1.5, 2] # ms
# wI_s_full = [550, 600, 650, 700, 750] # nS


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

headings_hz = headings*Hz
P_HEADING = PoissonGroup(N_TL2, rates=headings_hz[0,:], name='P_HEADING')

flow_hz = flow*Hz
P_FLOW = PoissonGroup(N_TN2, rates=flow_hz[0,:], name='P_FLOW')

#global CPU4_memory_stimulus
CPU_MEMORY_starting_value = 50 #Hz
CPU4_memory_stimulus = CPU_MEMORY_starting_value*np.ones((T,N_CPU4)) * Hz
P_CPU4_MEMORY = PoissonGroup(N_CPU4, rates=CPU4_memory_stimulus[0,:], name='P_CPU4_MEMORY')


# Neuron groups already optimised
G_TL2 = nc.generate_neuron_groups(N_TL2, eqs, threshold_eqs, reset_eqs, TL2_neuron_params, name='TL2')
G_CL1 = nc.generate_neuron_groups(N_CL1, eqs, threshold_eqs, reset_eqs, CL1_neuron_params, name='CL1')
G_TB1 = nc.generate_neuron_groups(N_TB1, eqs, threshold_eqs, reset_eqs, TB1_neuron_params, name='TB1')
G_TN2 = nc.generate_neuron_groups(N_TN2, eqs, threshold_eqs, reset_eqs, TN2_neuron_params, name='TN2')
G_CPU4 = nc.generate_neuron_groups(N_CPU4, eqs, threshold_eqs, reset_eqs, CPU4_neuron_params, name='CPU4')

print(f'TL2_neuron_params {TL2_neuron_params}')
print(f'CL1_neuron_params {CL1_neuron_params}')
print(f'TB1_neuron_params {TB1_neuron_params}')
print(f'TN2_neuron_params {TN2_neuron_params}')
print(f'CPU4_neuron_params {CPU4_neuron_params}')

# Neuron groups to optimise
G_CPU1A = nc.generate_neuron_groups(N_CPU1A, eqs, threshold_eqs, reset_eqs, neuron_params, name='CPU1A')
G_CPU1B = nc.generate_neuron_groups(N_CPU1B, eqs, threshold_eqs, reset_eqs, neuron_params, name='CPU1B')
G_PONTINE = nc.generate_neuron_groups(N_PONTINE, eqs, threshold_eqs, reset_eqs, neuron_params, name='PONTINE')
G_MOTOR = nc.generate_neuron_groups(N_MOTOR, eqs, threshold_eqs, reset_eqs, neuron_params, name='MOTOR')


# Synapses optimised
# Inputs
S_P_HEADING_TL2 = nc.connect_synapses(P_HEADING, G_TL2, W_HEADING_TL2, model=synapses_model, params=H_TL2_synapses_params, on_pre=synapses_eqs_ex, name='S_P_HEADING_TL2')
S_P_FLOW_TN2 = nc.connect_synapses(P_FLOW, G_TN2, W_FLOW_TN2, model=synapses_model, params=F_TN2_synapses_params, on_pre=synapses_eqs_ex, name='S_P_FLOW_TN2')

# TL2
S_TL2_CL1 = nc.connect_synapses(G_TL2, G_CL1, W_TL2_CL1, model=synapses_model, params=TL2_CL1_synapses_params, on_pre=synapses_eqs_ex, name='S_TL2_CL1')

# CL1
S_CL1_TB1 = nc.connect_synapses(G_CL1, G_TB1, W_CL1_TB1, model=synapses_model, params=CL1_TB1_synapses_params, on_pre=synapses_eqs_ex, name='S_CL1_TB1')

# TN2
S_TN2_CPU4 = nc.connect_synapses(G_TN2, G_CPU4, W_TN2_CPU4, model=synapses_model, params=TN2_CPU4_synapses_params, on_pre=synapses_eqs_ex, name='S_TN2_CPU4')

# TB1
S_TB1_TB1 = nc.connect_synapses(G_TB1, G_TB1, W_TB1_TB1, model=synapses_model, params=TB1_TB1_synapses_params, on_pre=synapses_eqs_in, name='S_TB1_TB1')
S_TB1_CPU4 = nc.connect_synapses(G_TB1, G_CPU4, W_TB1_CPU4, model=synapses_model, params=TB1_CPU4_synapses_params, on_pre=synapses_eqs_in, name='S_TB1_CPU4')



print(f'H_TL2_synapses_params {H_TL2_synapses_params}')
print(f'F_TN2_synapses_params {F_TN2_synapses_params}')
print(f'TL2_CL1_synapses_params {TL2_CL1_synapses_params}')
print(f'CL1_TB1_synapses_params {CL1_TB1_synapses_params}')
print(f'TN2_CPU4_synapses_params {TN2_CPU4_synapses_params}')
print(f'TB1_TB1_synapses_params {TB1_TB1_synapses_params}')
print(f'TB1_CPU4_synapses_params {TB1_CPU4_synapses_params}')

# Synapses to optimise
# TB1
S_TB1_CPU1A = nc.connect_synapses(G_TB1, G_CPU1A, W_TB1_CPU1A, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_in, name='S_TB1_CPU1A')
S_TB1_CPU1B = nc.connect_synapses(G_TB1, G_CPU1B, W_TB1_CPU1B, model=synapses_model, params=synapses_params,  on_pre=synapses_eqs_in, name='S_TB1_CPU1B')

# CPU4 accumulator
S_CPU4_M_PONTINE = nc.connect_synapses(P_CPU4_MEMORY, G_PONTINE, W_CPU4_PONTINE, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_PONTINE')
S_CPU4_M_CPU1A = nc.connect_synapses(P_CPU4_MEMORY, G_CPU1A, W_CPU4_CPU1A, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_CPU1A')
S_CPU4_M_CPU1B = nc.connect_synapses(P_CPU4_MEMORY, G_CPU1B, W_CPU4_CPU1B, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_CPU1B')

# Pontine
S_PONTINE_CPU1A = nc.connect_synapses(G_PONTINE, G_CPU1A, W_PONTINE_CPU1A, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_in, name='S_PONTINE_CPU1A')
S_PONTINE_CPU1B = nc.connect_synapses(G_PONTINE, G_CPU1B, W_PONTINE_CPU1B, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_in, name='S_PONTINE_CPU1B')

# CPU1A
S_CPU1A_MOTOR = nc.connect_synapses(G_CPU1A, G_MOTOR, W_CPU1A_MOTOR, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU1A_MOTOR')

# CPU1B
S_CPU1B_MOTOR = nc.connect_synapses(G_CPU1B, G_MOTOR, W_CPU1B_MOTOR, model=synapses_model, params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU1B_MOTOR')



#### Target

#CPU4_spike_rates = 100 # Hz
#CPU1A_spike_rates = 
#CPU1B_spike_rates
#MOTOR_spike_rates = 100 # HZ

# Scale spike rates from rate-based CX in the right range
# transpose since log is neuron_index*time_step but we want the opposite
#CPU4_stimulus = TimedArray(CPU4_spike_rates*cx_log.cpu4.T*Hz, dt=1.*time_step*ms)
#P_CPU4 = PoissonGroup(N_CPU4, rates='CPU4_stimulus(t,i)')

# second to penultimate rows are CPU1A
# CPU1A_rate = cx_log.cpu1[1:-1,:].T
# CPU1A_stimulus = TimedArray(CPU1A_spike_rates*CPU1A_rate*Hz, dt=1.*time_step*ms)
# P_CPU1A = PoissonGroup(N_CPU1A, rates='CPU1A_stimulus(t,i)')

# # first and last rows are CPU1B
# CPU1B_rate = cx_log.cpu1[[0,-1],:].T
# CPU1B_stimulus = TimedArray(CPU1B_spike_rates*CPU1B_rate*Hz, dt=1.*time_step*ms)
# P_CPU1B = PoissonGroup(N_CPU1B, rates='CPU1B_stimulus(t,i)')

# PONTINE_stimulus = TimedArray(TB1_spike_rates*cx_log.PONTINE.T*Hz, dt=1.*time_step*ms)
# P_PONTINE = PoissonGroup(N_PONTINE, rates='PONTINE_stimulus(t,i)')

motors = cx_spiking.inputs.compute_motors(cx_log.cpu1)
MOTOR_stimulus = TimedArray(40*motors.T*Hz, dt=1.*time_step*ms)
P_MOTOR = PoissonGroup(N_MOTOR, rates='MOTOR_stimulus(t,i)')


global CPU4_memory, CPU4_memory_history
CPU4_memory_history = CPU_MEMORY_starting_value * np.ones((T, N_CPU4))
CPU4_memory = CPU_MEMORY_starting_value * np.ones(N_CPU4)


def get_agent_timestep(t, sim_time_step):
    return int((t/ms + 0.5) / sim_time_step)


def extract_spike_counts(SPM, t, time_step):
    '''
    Count spikes for each neuron in the [t-time_step, t] interval
    of the simulation
    '''
    spike_trains = SPM.spike_trains()
    neurons = np.zeros(len(SPM.spike_trains()), dtype=int)
    for idx in range(len(spike_trains)):
        spike_train = spike_trains[idx]
        neurons[idx] = len(spike_train[(spike_train > t-time_step*ms) & (spike_train < t)])
    return neurons


@network_operation(dt=time_step*ms, when='start', order=2, name='CPU4_accumulator')
def CPU4_accumulator(t):
    global CPU4_memory, CPU4_memory_history, CPU4_memory_stimulus
    
    timestep = get_agent_timestep(t, time_step)
    
    if t < time_step*ms:
        return

    neurons_responses = G_CPU4.spike_count

    mem_update = neurons_responses 
    CPU4_memory = CPU4_memory_history[timestep-1,:]
    CPU4_memory += mem_update * 0.05
    CPU4_memory -= 0.025 * (1./(mem_update+0.1))
    CPU4_memory = np.clip(CPU4_memory, 0, np.inf)
    CPU4_memory_history[timestep,:] = CPU4_memory

    
    CPU4_memory_stimulus[timestep,:] = CPU4_memory * Hz
    
    G_CPU4.spike_count = 0


######################################
### NETWORK OPERATIONS
######################################
global ref_angles, heading_angles, velocities 

ref_angles = np.linspace(-np.pi+np.pi/8, np.pi+np.pi/8, N_TB1, endpoint=False)
max_velocity = 12 

heading_angles = np.zeros(T)
velocities = np.zeros((T, 2))

def circular_weighted_mean(weights, angles):
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight
    mean = math.degrees(math.atan2(y, x))
    return mean


def make_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def compute_peak(neurons_responses, ref_angles):
    # trick to get the correct weighted average of where the heading is
    # create a list with all the angles between [-pi,pi] repeated by their count
    # so [0,2,0,0,1,0,0,1] will be [-1.963, -1.963, 0.392, 2.748] and then compute
    # circular mean between [-pi, pi]
    tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons_responses[i])]
    # -pi/8 because we center the neurons at the center of their pi/4 receptive fields
    peak = scipy.stats.circmean(tmp, low=-np.pi, high=np.pi) - np.pi/8
    return make_angle(peak)


@network_operation(dt=time_step*ms, when='start', order=0, name='extract_heading')
def extract_heading(t):
    global ref_angles, heading_angles

    timestep = get_agent_timestep(t, time_step)
    
    if t < time_step*ms:
        heading_angles[timestep] = compute_peak([0,0,0,1,1,0,0,0], ref_angles)
        #neurons = [0,0,0,1,1,0,0,0]
        #tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons[i])]
        #heading_angles[timestep] = make_angle(scipy.stats.circmean(tmp, low=-np.pi, high=np.pi))
        G_TB1.spike_count = 0
        return

    neurons_responses = G_TB1.spike_count

    if np.sum(neurons_responses) > 0:
        #tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons_responses[i])]
        # -pi/8 because we center the neurons at the center of their pi/4 receptive fields
        #peak = scipy.stats.circmean(tmp, low=-np.pi, high=np.pi) - np.pi/8
        #heading_angles[timestep] = make_angle(peak)
        heading_angles[timestep] = compute_peak(neurons_responses, ref_angles)
    else:
        heading_angles[timestep] = heading_angles[timestep-1]
    
    G_TB1.spike_count = 0


@network_operation(dt=time_step*ms, when='start', order=1, name='extract_velocity')
def extract_velocity(t):
    global velocities, max_velocity
    
    timestep = get_agent_timestep(t, time_step)

    if t < time_step*ms:
        velocities[timestep] = [0,0]
        G_TN2.spike_count = 0
        return
    neurons_responses = G_TN2.spike_count

    neurons_responses = np.clip(neurons_responses, 0, max_velocity)
    velocities[timestep] = neurons_responses / max_velocity

    G_TN2.spike_count = 0
    

new_heading_dir = np.zeros(T)
new_velocities = np.zeros((T,2))

def get_next_velocity(heading, velocity, rotation, acceleration=0.3, drag=0.15):
    def thrust(theta, acceleration):
        return np.array([np.sin(theta), np.cos(theta)]) * acceleration
    v = velocity + thrust(heading, acceleration).flatten()
    v -= drag * v
    return np.clip(v, 0, np.inf)

@network_operation(dt=time_step*ms, when='start', order=3, name='update_inputs')  
def update_inputs(t):
    timestep = get_agent_timestep(t, time_step)

    #### motor response - rotation
    #print(extract_spike_counts(SPM_MOTOR, (sim_timestep)*time_step*ms, time_step), G_MOTOR.spike_count)
    motor_responses = G_MOTOR.spike_count
    rotation = np.sign(motor_responses[0] - motor_responses[1])
    #print(motor_responses, rotation)
    G_MOTOR.spike_count = 0
    
    #### heading
    # previous heading
    prev_heading = np.array([heading_angles[timestep]])
    # compute spikes based on old heading and rotation using fixed angle "step" of 22.5 degrees 
    new_heading = prev_heading + rotation * 0.008 # mean and median rotation found from rate model
    new_heading_dir[timestep] = new_heading
    new_headings = cx_spiking.inputs.compute_headings(new_heading, N=N_TL2//2, vmin=5, vmax=100)
    new_headings = np.tile(new_headings, 2) 
    # save new heading
    headings_hz[timestep,:] = new_headings * Hz

    
    #### velocity
    velocity = np.array(velocities[timestep,:])
    updated_v = get_next_velocity(new_heading, velocity, rotation)
    new_velocities[timestep,:] = updated_v
    new_flow = cx_spiking.inputs.compute_flow(new_heading, updated_v, baseline=50, 
                                              vmin=0, vmax=50, inbound=True)
    flow_hz[timestep,:] = new_flow * Hz


@network_operation(dt=time_step*ms, when='start', order=4, name='set_rates')
def set_rates(t):
    timestep = get_agent_timestep(t, time_step)

    if t < time_step*ms:
        return
    P_HEADING.rates = headings_hz[timestep,:]
    P_FLOW.rates = flow_hz[timestep,:]
    P_CPU4_MEMORY.rates = CPU4_memory_stimulus[timestep,:]

net = Network(collect())
net['update_inputs'].active = False

net.store('initialised')

######################################
### OPTIMISER
######################################
def run_simulation_NET(net, cx_log, tauE_, wE_, tauI_, wI_, 
                           Group, Target,
                           G_CPU1A, G_CPU1B, G_PONTINE, G_MOTOR,
                           S_TB1_CPU1A,                  
                           S_CPU4_M_PONTINE,
                           S_CPU4_M_CPU1A,
                           S_PONTINE_CPU1A, 
                           S_TB1_CPU1B,
                           S_CPU4_M_CPU1B, 
                           S_PONTINE_CPU1B, 
                           S_CPU1A_MOTOR, 
                           S_CPU1B_MOTOR,  
                           time, dt_, delta, rate_correction): 

    net.restore('initialised') 

    # set the parameters 
    Group.set_states({'tauE' : tauE_*ms, 'tauI' : tauI_*ms})
    print(f'taueE: {tauE_} - tauI {tauI_}')
    G_CPU1A.set_states({'tauE' : tauE_*ms, 'tauI' : tauI_*ms})
    G_CPU1B.set_states({'tauE' : tauE_*ms, 'tauI' : tauI_*ms})
    G_PONTINE.set_states({'tauE' : tauE_*ms, 'tauI' : tauI_*ms})
    G_MOTOR.set_states({'tauE' : tauE_*ms, 'tauI' : tauI_*ms})

    S_TB1_CPU1A.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_CPU4_M_PONTINE.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_CPU4_M_CPU1A.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_PONTINE_CPU1A.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_TB1_CPU1B.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_CPU4_M_CPU1B.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_PONTINE_CPU1B.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_CPU1A_MOTOR.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})
    S_CPU1B_MOTOR.set_states({'wE' : wE_*nS, 'wI' : wI_*nS})

    print(f'wE: {wE_} - wI {wI_}')

    
    _, model_spike_monitor_motor = nc.add_monitors(Group)
    target_spike_monitor_motor = SpikeMonitor(Target, name='MOTOR_target_spike_monitor')

    net.run(time)

    gf_motor = metric.compute_gamma_factor(model_spike_monitor_motor, target_spike_monitor_motor, time, 
                                     dt_=dt_, delta=delta, rate_correction=rate_correction)

    motors_rate_model = cx_spiking.inputs.compute_motors(cx_log.cpu1)
    rotations_rate_model = np.sign(motors_rate_model[0,:1500]-motors_rate_model[1,:1500])

    MOTOR_spikes =  cx_spiking.inputs.get_spikes_rates(target_spike_monitor_motor, 2, 1500, 20)
    rotations_spike_model = np.sign((MOTOR_spikes[0,:1500]-MOTOR_spikes[1,:1500]))

    measure = np.sum(np.abs(rotations_rate_model-rotations_spike_model))

    print(f'Gamma factor: {gf_motor} - {measure}')

    gf = gf_motor
    return gf



tauE_s_full = [0.5, 1, 1.5, 2, 2.5] # ms
wE_s_full = [300, 400, 500, 600, 700, 800] # nS
tauI_s_full = [0.5, 1, 1.5, 2, 2.5] # ms
wI_s_full = [400, 500, 600, 700, 800, 900] # nS


gamma_factors = np.zeros((len(tauE_s_full), len(wE_s_full), len(tauI_s_full), len(wI_s_full)))

delta = 1*ms
rate_correction = True


tauE_ = tauE_s_full[args.index]
for we_, wE_ in enumerate(wE_s_full):
    for ti_, tauI_ in enumerate(tauI_s_full):
        for wi_, wI_ in enumerate(wI_s_full):
            gamma_factors[args.index, we_, ti_, wi_] = run_simulation_NET(net, cx_log,
                                                                          tauE_, wE_, tauI_, wI_, 
                                                                          G_MOTOR, P_MOTOR,
                                                                          G_CPU1A, G_CPU1B, G_PONTINE, G_MOTOR, 
                                                                          S_TB1_CPU1A,                  
                                                                          S_CPU4_M_PONTINE,
                                                                          S_CPU4_M_CPU1A,
                                                                          S_PONTINE_CPU1A, 
                                                                          S_TB1_CPU1B,
                                                                          S_CPU4_M_CPU1B, 
                                                                          S_PONTINE_CPU1B, 
                                                                          S_CPU1A_MOTOR, 
                                                                          S_CPU1B_MOTOR,  
                                                                          T_outbound*time_step*ms, 
                                                                          defaultclock.dt, delta, rate_correction)

with open(f'outputs/NET_gamma_factors_grid_search_{args.index}.npz', 'wb') as f:
    np.save(f, gamma_factors)
#np.savetxt('outputs/TB1_gamma_factors_grid_search.csv', gamma_factors, delimiter=',')

candidate = np.argwhere(gamma_factors == np.min(gamma_factors))[0]

print('Final Candidate')
print(candidate, gamma_factors[candidate[0], candidate[1], candidate[2], candidate[3]])
print(tauE_s_full[candidate[0]], wE_s_full[candidate[1]], tauI_s_full[candidate[2]], wI_s_full[candidate[3]])


print('***********  DONE  ***********')