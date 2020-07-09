import numpy as np
from brian2 import *


def compute_gamma_factor(exp_monitor, target_monitor, time, 
                         dt=defaultclock.dt, delta=1*ms, rate_correction=True):
    r'''
    Return mean gamma factor

    Parameters
    ----------
    exp_monitor: SpikeMonitor
        spike monitor of the model population
    target_monitor: SpikeMonitor
        spike monitor of the target population
    time: brain2 ms
        time of simulation in milliseconds
    dt: brain2 ms
        timestep of simulation
    delta: brain2 ms
        time window for coincident spikes
    rate_correction: boolean (True)
        if True the error term is more accurate 
        check brian2modelfitting docs for more details

    Returns
    -------
    mean gamma factor: scalar
        average of gamma factor for each neuron in the population
    '''

    input_spikes = get_spikes(exp_monitor)
    output_spikes = get_spikes(target_monitor)

    all_gf = []
    for model, target in zip(input_spikes, output_spikes):
        gf = get_gamma_factor(model, target, delta, time, dt, 
                              rate_correction=rate_correction)
        all_gf.append(gf)

    return np.array(all_gf).mean()


def get_spikes(monitor):
    r'''
    Return list of lists with spike times

    Parameters
    ----------
    monitor: SpikeMonitor
        monitor of the neural population
    
    Returns
    -------
    spikes: list of list
        list of spike times for each neuron
    '''
    N = len(monitor.count)
    spikes_t = np.array(monitor.t)
    spikes = [spikes_t[monitor.i == i] for i in range(N)]
    return spikes


def get_gamma_factor(model, data, delta, time, dt, rate_correction=True):
    r'''
    Calculate gamma factor between model and target spike trains,
    with precision delta.

    Parameters
    ----------
    model: `list` or `~numpy.ndarray`
        model trace
    data: `list` or `~numpy.ndarray`
        data trace
    delta: `~brian2.units.fundamentalunits.Quantity`
        time window
    dt: `~brian2.units.fundamentalunits.Quantity`
        time step
    time: `~brian2.units.fundamentalunits.Quantity`
        total time of the simulation
    rate_correction: bool
        Whether to include an error term that penalizes differences in firing
        rate, following `Clopath et al., Neurocomputing (2007)
        <https://doi.org/10.1016/j.neucom.2006.10.047>`_.

    Returns
    -------
    float
        An error based on the Gamma factor. If ``rate_correction`` is used,
        then the returned error is :math:`1 + 2\frac{\lvert r_\mathrm{data} - r_\mathrm{model}\rvert}{r_\mathrm{data}} - \Gamma`
        (with :math:`r_\mathrm{data}` and :math:`r_\mathrm{model}` being the
        firing rates in the data/model, and :math:`\Gamma` the coincidence
        factor). Without ``rate_correction``, the error is
        :math:`1 - \Gamma`. Note that the coincidence factor :math:`\Gamma`
        has a maximum value of 1 (when the two spike trains are exactly
        identical) and a value of 0 if there are only as many coincidences
        as expected from two homogeneous Poisson processes of the same rate.
        It can also take negative values if there are fewer coincidences
        than expected by chance.
    '''
    model = np.array(model)
    data = np.array(data)

    model = np.array(np.rint(model / dt), dtype=int)
    data = np.array(np.rint(data / dt), dtype=int)
    delta_diff = int(np.rint(delta / dt))

    model_length = len(model)
    data_length = len(data)
    # data_rate = firing_rate(data) * Hz
    data_rate = data_length / time
    model_rate = model_length / time

    if model_length > 1:
        bins = .5 * (model[1:] + model[:-1])
        indices = np.digitize(data, bins)
        diff = np.abs(data - model[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = np.sum(matched_spikes)
    elif model_length == 0:
        coincidences = 0
    else:
        indices = [np.amin(np.abs(model - data[i])) <= delta_diff for i in np.arange(data_length)]
        coincidences = np.sum(indices)

    # Normalization of the coincidences count
    NCoincAvg = 2 * delta * data_length * data_rate
    norm = .5*(1 - 2 * data_rate * delta)
    gamma = (coincidences - NCoincAvg)/(norm*(model_length + data_length))

    if rate_correction:
        rate_term = 1 + 2*np.abs((data_rate - model_rate)/data_rate)
    else:
        rate_term = 1

    return np.clip(rate_term - gamma, 0, np.inf)
