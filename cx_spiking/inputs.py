import os

import numpy as np
import scipy

from brian2 import * 

import trials
from cx_spiking.constants import *

def generate_route(T_outbound, vary_speed, route_file='', load_route=True):
    if load_route and os.path.exists(route_file):
        print(f'Load route from {route_file}')
        with np.load(route_file) as data:
            h = data['h']
            v = data['v']
    else:
        print(f'Generating route of length {T_outbound} and vary_speed={vary_speed}')
        h, v = trials.generate_route(T=T_outbound, vary_speed=vary_speed)

    return h, v


def save_route(route_file, h, v, save_route=True):
    if (os.path.exists(route_file)):
        print(f'{route_file} exists - not overwriting it')
    elif save_route and not os.path.exists(route_file):
        print("Saving route...")
        np.savez_compressed(route_file, h=h, v=v)

def normalise_range(data, vmin=5, vmax=100):
    d = np.where(data >= 0, data, 0)
    d = ((d - d.min()) * (vmax - vmin)) / (d.max() - d.min()) + vmin
    return d
    

def compute_headings(h, N=8, sigma=0.8, vmin=5, vmax=100):
    r'''
    Convert headings from Stone's simulator to spike rates
    that can be fed into a brian2 PoissonGroup

    Parameters
    ----------
    h: headings
        headings generated from Stone's simulator
    N: number of directions
        8 representing the cardinal directions (N, NE, E, SE, S, SW, W, NW)
    sigma: standard deviation 
        of a Gaussian distribution
    vmin: hertz
        minimum value for rescaling the rates
    vmax: hertz
        maximum value for rescaling the rates

    Returns
    -------
    headings: numpy array(n_steps * n_neurons)
        each row represents the rate for the neurons
    '''
    T_outbound = h.shape[0]
    headings = np.zeros((len(h), N))

    # Normal(mu, sigma) is equivalent to vonMises(mu, 1./sigma**2)
    kappa = 1. / sigma**2
    # Fixed locations of angles
    x = np.linspace(0-np.pi, 0+np.pi, N+1, endpoint=True)

    for step in range(T_outbound):
        samples = scipy.stats.vonmises.pdf(x, kappa, loc=h[step])
        headings[step,:] = np.roll(samples[:N], -int(np.ceil(N/2)))

    # Normalize between 5-100 Hz, the headings represents rate
    if vmin >= 0 and vmax > 0 and vmax > vmin:
        headings = normalise_range(headings, vmin=vmin, vmax=vmax)

    return headings


def compute_headings_old(h, N=8, loc=0, scale=0.8, vmin=5, vmax=100):
    T_outbound = h.shape[0]

    rv = scipy.stats.norm(loc=loc, scale=scale)
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), N, endpoint=True)
    pdf = rv.pdf(x)

    # Split the [-pi,pi] interval into bins
    bins = np.linspace(-np.pi, np.pi, N+1)

    # -1 required since the returned bins are 1-indexed
    digitized = np.digitize(h, bins) - 1
    
    headings = np.zeros((len(h), N))

    for t in range(T_outbound):
        # Shift angle
        # angle_shift = N//2 + digitized[t]
        angle_shift = digitized[t]
        headings[t,:] = np.roll(pdf, angle_shift)
    
    # Normalize between 5-100 Hz, the headings represents rate
    if vmin >= 0 and vmax > 0 and vmax > vmin:
        headings = normalise_range(headings, vmin=vmin, vmax=vmax)
    
    return headings, digitized    


def compute_flow(heading, velocity, preferred_angle=np.pi/4, baseline=50, vmin=0, vmax=50):
    r'''
    Calculate optic flow based on a preferred angle of +-45 degrees
    Set a baseline and add the rescaled values between [vmin,vmax]

    Code from Stone's - central_complex.py, line 45

    Parameters
    ----------
    heading: headings
        headings generated from Stone's simulator
    velocity: velocity
        velocity generated from Stone's simulator
    preferred_angle: radians
        preferred angle for optic flow, +- around 0
    baseline: hertz 
        baseline for the velocity rates
    vmin: hertz
        minimum value for rescaling the rates
    vmax: hertz
        maximum value for rescaling the rates

    Returns
    -------
    flow: numpy array(n_steps * 2)
        values are in [baseline + vmin, baseline + vmax]
        each row is stored as [L, R]
    '''
    flow = np.zeros((velocity.shape[0], 2))
    for i in range(velocity.shape[0]):
        A = np.array([[np.sin(heading[i] + preferred_angle),
                       np.cos(heading[i] + preferred_angle)],
                      [np.sin(heading[i] - preferred_angle),
                       np.cos(heading[i] - preferred_angle)]])
        flow[i,:] = np.dot(A, velocity[i,:])
    # Clip in [0,1] as in Stone et al.
    flow = np.clip(flow, 0, 1)
    
    if vmin >= 0 and vmax > 0 and vmax > vmin:
        flow = normalise_range(flow, vmin=vmin, vmax=vmax)

    flow = flow + baseline
    return flow


def get_spikes_rates(SPM, N, T_outbound, time_step):
    spikes_t = SPM.t/ms
    spikes_i = SPM.i
    
    spikes_out = np.zeros((N, T_outbound))
    bins = np.arange(0, (T_outbound+1)*time_step, time_step)
    
    for i in range(N):
        spikes = spikes_t[spikes_i == i]
        spikes_count, _ = np.histogram(spikes, bins=bins)
        spikes_out[i,:] = spikes_count

    return spikes_out


def to_Hertz(data, time_step):
    # convert milliseconds to seconds
    return data/(time_step / 1000)


def compute_motors(cpu1):
    cpu1a = cpu1[1:-1]
    cpu1b = np.array([cpu1[-1], cpu1[0]])
    motor = np.dot(W_CPU1A_MOTOR, cpu1a)
    motor += np.dot(W_CPU1B_MOTOR, cpu1b)
    # Consistent with Stone's code
    motor = motor[[1,0],:]
    return motor
