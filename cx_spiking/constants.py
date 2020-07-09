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


###############################################################################
###                             EQUATIONS
###############################################################################

# Neuron specification
EL = -52 * mV # resting potential (mV)
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
      '''

threshold_eqs = 'Vm >= Vt'
reset_eqs = 'Vm = Vr'


params = {'EL' : [EL],
          'Vm' : [EL],
          'EE' : [0*mV],
          'EI' : [-80*mV],
          'gL' : [1*10**-6 * siemens],
          'gE' : '(randn() * 1.5 + 4) * 10.*nS',
          'gI' : '(randn() * 12 + 20) * 10.*nS',
          #'tauE' : [1*ms],
          #'tauI' : [2*ms]
         }


synapses_model = '''w : 1'''
synapses_eqs_ex = '''gE += wE * w'''
synapses_eqs_in = '''gI += wI * w'''


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
W_TL2_CL1 = np.eye(N_TL2)
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