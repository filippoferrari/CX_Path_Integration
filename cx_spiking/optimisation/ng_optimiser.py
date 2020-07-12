import numpy as np
import nevergrad as ng


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