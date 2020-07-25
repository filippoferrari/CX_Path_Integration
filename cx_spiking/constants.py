import numpy as np
from brian2 import *



###############################################################################
###                             CONSTANTS
###############################################################################
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN2 = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B
N_PONTINE = 16

N_MOTOR = 2

tauE_s = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # ms
wE_s = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # nS
tauI_s = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] # ms
wI_s = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000] # nS



tauE_s_full = [0.5, 1, 1.5, 2, 2.5] # ms
wE_s_full = [300, 400, 500, 600, 700, 800] # nS
tauI_s_full = [0.5, 1, 1.5, 2, 2.5] # ms
wI_s_full = [400, 500, 600, 700, 800, 900] # nS




TL2_spike_rates = 90 # Hz

TN2_spike_rates_min = 50 # Hz
TN2_spike_rates_max = 160 # Hz

CL1_spike_rates = 50 # Hz

TB1_spike_rates = 50 # Hz


###############################################################################
###                             EQUATIONS
###############################################################################

#### Neuron specification
#EL = -52 * mV # resting potential (mV)
Vt = -45 * mV # spike threshold (mV)
Vr = -52 * mV # reset potential (mV)

Cm = 0.002 * ufarad # membrane capacitance (uF)
Rm = 10 * Mohm      # membrane resistance (MOhm)
taum = Cm * Rm      # = 20ms membrane time constant (ms) 


eqs = '''
      dVm/dt = (gL * (EL - Vm) + gE * (EE - Vm) + gI * (EI - Vm)) / Cm : volt
      dgE/dt = -gE/tauE : siemens
      dgI/dt = -gI/tauI : siemens
      EL : volt
      EE : volt
      EI : volt
      gL : siemens
      tauE : second
      tauI : second
      '''

threshold_eqs = 'Vm >= Vt'
reset_eqs = 'Vm = Vr'


#### Synapses
synapses_eqs_ex = '''gE += wE * w'''
synapses_eqs_in = '''gI += wI * w'''

synapses_model = '''
                 w : 1
                 wE : siemens
                 wI : siemens
                 '''

###############################################################################
###                             PARAMETERS
###############################################################################
### Default 
neuron_params = {
    'EL' : [-52 * mV],
    'Vm' : [-52 * mV],
    'EE' : [0 * mV],
    'EI' : [-80 * mV],
    'gL' : [1*10**-6 * siemens],
    'gE' : '(randn() * 1.5 + 4) * 10.*nS',
    'gI' : '(randn() * 12 + 20) * 10.*nS',
    'tauE' : [1 * ms],
    'tauI' : [2 * ms]
}

synapses_params = {
    'wE' : [200 * nS],
    'wI' : [200 * nS]
}


### TL2
TL2_neuron_params = {
    'EL' : [-52 * mV],
    'Vm' : [-52 * mV],
    'EE' : [0 * mV],
    'EI' : [-80 * mV],
    'gL' : [1*10**-6 * siemens],
    'gE' : '(randn() * 1.5 + 4) * 10.*nS',
    'gI' : '(randn() * 12 + 20) * 10.*nS',
    'tauE' : [0.5 * ms],
    'tauI' : [2 * ms] # default
}

H_TL2_synapses_params = {
    'wE' : [900 * nS],
    'wI' : [200 * nS] # default
}


### TN2
TN2_neuron_params = {
    'EL' : [-52 * mV],
    'Vm' : [-52 * mV],
    'EE' : [0 * mV],
    'EI' : [-80 * mV],
    'gL' : [1*10**-6 * siemens],
    'gE' : '(randn() * 1.5 + 4) * 10.*nS',
    'gI' : '(randn() * 12 + 20) * 10.*nS',
    'tauE' : [3.5 * ms],
    'tauI' : [2 * ms] # default
}

F_TN2_synapses_params = {
    'wE' : [250 * nS],
    'wI' : [200 * nS] # default
}


### CL1
CL1_neuron_params = {
    'EL' : [-52 * mV],
    'Vm' : [-52 * mV],
    'EE' : [0 * mV],
    'EI' : [-80 * mV],
    'gL' : [1*10**-6 * siemens],
    'gE' : '(randn() * 1.5 + 4) * 10.*nS',
    'gI' : '(randn() * 12 + 20) * 10.*nS',
    'tauE' : [1.5 * ms],
    'tauI' : [2 * ms] # default
}

TL2_CL1_synapses_params = {
    'wE' : [450 * nS],
    'wI' : [200 * nS] # default
}


### TB1
TB1_neuron_params = {
    'EL' : [-52 * mV],
    'Vm' : [-52 * mV],
    'EE' : [0 * mV],
    'EI' : [-80 * mV],
    'gL' : [1*10**-6 * siemens],
    'gE' : '(randn() * 1.5 + 4) * 10.*nS',
    'gI' : '(randn() * 12 + 20) * 10.*nS',
    'tauE' : [0.5 * ms],
    'tauI' : [0.5 * ms]
}

CL1_TB1_synapses_params = {
    'wE' : [550 * nS],
    'wI' : [200 * nS] # default
}

TB1_TB1_synapses_params = {
    'wE' : [700 * nS], # default
    'wI' : [550 * nS] 
}

###############################################################################
###                             CONNECTIVITY MATRICES
###############################################################################


def gen_TB1_TB1_weights(weight=1.):
    """
    Weight matrix to map inhibitory connections from TB1 to other neurons
    
    from Thomas Stone's path-integration repo
    """
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W

W_HEADING_TL2 = np.eye(N_TL2)
W_FLOW_TN2 = np.eye(N_TN2)
# Act as if CL1 cells were inverting TL2 output
# by shifting it by 180 degrees 
W_TL2_CL1 = np.roll(np.eye(N_TL2),4, axis=1)
W_CL1_TB1 = np.tile(np.eye(N_TB1), 2)
W_TB1_TB1 = gen_TB1_TB1_weights()
W_TB1_CPU1A = np.tile(np.eye(N_TB1), (2, 1))[1:N_CPU1A+1, :]
W_TB1_CPU1B = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0, 0, 0, 0]])
W_TB1_CPU4 = np.tile(np.eye(N_TB1), (2, 1))
W_TN2_CPU4 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
]).T
W_CPU4_CPU1A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])
W_CPU4_CPU1B = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
])
W_PONTINE_CPU1A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15
])
W_PONTINE_CPU1B = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #9
])
W_CPU4_PONTINE = np.eye(N_CPU4)

# Not sure about these
W_CPU1A_MOTOR = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
])
W_CPU1B_MOTOR = np.array([[0, 1],
                          [1, 0]])