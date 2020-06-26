import numpy as np
from brian2 import *

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B
N_PONTINE = 2

class CX(object):
    '''
    Class for a Central Complex object
    '''

    def __init__(self, 
                 neuron_eqs,
                 threshold_eqs,
                 res):
        pass