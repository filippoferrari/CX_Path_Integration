import numpy as np
import matplotlib.pyplot as plt

from brian2 import *


def plot_stuff(M_spikes, M, name='', observation_list=[0], figsize=(10,7)):
    figure(figsize=figsize)
    #plotting spikes of cells on raster plot.
    subplot(2,2,1)
    title(f'{name} spikes')
    plot(M_spikes.t/ms, M_spikes.i, '.k')
    
    #observation_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    observation_list = observation_list

    #plotting voltage
    subplot(2,2,2)
    if 'Vm' in M.variables:
        title('Vm')
        for i in observation_list:
            plot(M.t/ms, M.Vm[i])

    subplot(2,2,3)
    if 'ge' in M.variables:
        title('ge')
        for i in observation_list:
            plot(M.t/ms, M.ge[i])

    subplot(2,2,4)
    if 'gi' in M.variables:
        title('gi')
        for i in observation_list:
            plot(M.t/ms, M.gi[i])

    show()


def visualise_connectivity(S, plot_name=''):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(14, 4))
    suptitle(plot_name)
    
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

    #subplot(133)
    #scatter(S.x_pre/um, S.x_post/um, S.w*20)
    #xlabel('Source neuron position (um)')
    #ylabel('Target neuron position (um)')
    show()


def plot_connectivity_matrix(W_matrix, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    s = ax.pcolor(W_matrix, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_aspect('equal')
    cax = fig.add_axes([1.02, 0.05, 0.02, 0.9])
    fig.colorbar(s, ax=ax, cax=cax)
    plt.show()