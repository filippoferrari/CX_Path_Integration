import numpy as np
from brian2 import *

import cx_spiking.plotting


def generate_neuron_groups(N, eqs, threshold_eqs, reset_eqs, 
                           params, name='', method='euler'):
    ng = NeuronGroup(N=N, model=eqs, threshold=threshold_eqs, reset=reset_eqs, method='euler', name=name)
    ng.set_states(params)
    return ng


def add_monitors(N_group, variables_to_record=['Vm', 'gE', 'gI'], name=''):
    state_monitor = StateMonitor(N_group, variables_to_record, record=True, name=f'{name}_stm')
    spike_monitor = SpikeMonitor(N_group, name=f'{name}_spm')
    return state_monitor, spike_monitor


def connect_synapses(G_source, G_target, W_matrix, model, params, on_pre, plot_name=None):
    # The connectivity matrix defined by Thomas Stone has sources on 
    # the rows and targets on the columns whereas Brian2 expects 
    # the opposite So you need to transpose it!
    sources, targets = W_matrix.T.nonzero()
    synapses = Synapses(G_source, G_target, model=model, on_pre=on_pre)
    synapses.connect(i=sources, j=targets)
    synapses.w = W_matrix.T[sources, targets]

    synapses.set_states(params)

    if plot_name:
        print(sources)
        print(targets)
        cx_spiking.plotting.visualise_connectivity(synapses, plot_name=plot_name)
        
    return synapses

