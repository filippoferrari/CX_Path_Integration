import os

import numpy as np
import scipy

from brian2 import * 

import trials

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
    

def compute_headings(h, N=8, loc=0, scale=0.8, vmin=5, vmax=100):
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
        angle_shift = N//2 + digitized[t]
        headings[t,:] = np.roll(pdf, angle_shift)
    
    # Normalize between 5-100 Hz, the headings represents rate
    if vmin >= 0 and vmax > 0 and vmax > vmin:
        headings = normalise_range(headings, vmin=vmin, vmax=vmax)
    
    return headings, digitized    


def compute_flow(heading, velocity, baseline=50, vmin=0, vmax=50, preferred_angle=np.pi/4):
    '''
    Calculate optic flow depending on preference angles. [L, R]
    Preferred angle is 45 degrees
    From central_complex.py, line 45
    Baseline is 50 Hz
    Flow values are in [baseline + vmin, baseline + vmax]
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