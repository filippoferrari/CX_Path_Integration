import numpy as np
import nevergrad as ng

# from brian2 import *

# import cx_spiking.optimisation.metric as metric
# import cx_spiking.network_creation as nc

def set_instrumentation(bounds, args):
    instruments = []
    for bound in range(len(bounds)):
        assert len(bounds[bound]) == 2
        instrumentation = ng.instrumentation.var.Array(1).asscalar().bounded(np.array([bounds[bound][0]]),
                                                                             np.array([bounds[bound][1]]))
        instruments.append(instrumentation)

    for arg in range(len(args)):
        instruments.append(args[arg])
    print(instruments)

    instrum = ng.instrumentation.Instrumentation(*instruments)

    return instrum


def set_optimiser(instruments, method='DE', budget=100):
    optim = ng.optimization.registry[method](instrumentation=instruments, budget=budget)
    return optim


def print_candidate_and_value(optimizer, candidate, value):
    print(candidate, value)


def run_optimiser(optim, function, verbosity=0):
    optim.register_callback("tell", print_candidate_and_value)
    recommendation = optim.minimize(function, verbosity=verbosity)  # best value
    return optim, recommendation


def run_optimisers(instruments, function, methods=['DE'], budget=100, verbosity=0):
    out = {}
    for method in methods:
        optim = set_optimiser(instruments, method=method, budget=budget)
        optim_min, recommendation = run_optimiser(optim, function, verbosity=verbosity)
        out[method] = (optim_min, recommendation)
    return out



'''
Tried to move functions here, but too messy
keep them in the same context of the script
'''
# def run_simulation_TL2(tauE_, wE_, tauI_, wI_, network, Group, Synapses, Target,
#                        time, dt_, delta, rate_correction): 
#     network.restore('initialised') 

#     # set the parameters 
#     Group.set_states({'tauE' : tauE_*ms,
#                       'tauI' : tauI_*ms})
#     print(f'taueE: {tauE_} - tauI {tauI_}')

#     Synapses.set_states({'wE' : wE_*nS,
#                          'wI' : wI_*nS})
#     print(f'wE: {wE_} - wI {wI_}')

#     print(Group)
#     print(Synapses)
#     print(Target)
    
#     _, model_spike_monitor = nc.add_monitors(Group)
#     target_spike_monitor = SpikeMonitor(Target, name='TL2_target_spike_monitor')
#     print(target_spike_monitor)
    
#     run(time)

#     gf = metric.compute_gamma_factor(model_spike_monitor, target_spike_monitor, time, 
#                                      dt_=dt_, delta=delta, rate_correction=rate_correction)
    
#     print(f'Gamma factor: {gf}')

#     return gf