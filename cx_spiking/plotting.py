import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
from brian2tools import *

def plot_stuff(STM, SPM, name='', observation_list=[0], Vt=-0.045, figsize=(10,7), savefig_=None):
    figure(figsize=figsize)
    #plotting spikes of cells on raster plot.
    subplot(2,2,1)
    title(f'{name} spikes')
    plot(SPM.t/ms, SPM.i, '.k')
    
    #observation_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    observation_list = observation_list

    #plotting voltage
    subplot(2,2,2)
    if 'Vm' in STM.variables:
        title('Vm')
        for i in observation_list:
            plot(STM.t/ms, STM.Vm[i])
        axhline(Vt, ls='--', alpha=0.4)

    subplot(2,2,3)
    if 'ge' in STM.variables:
        title('ge')
        for i in observation_list:
            plot(STM.t/ms, STM.ge[i])
    elif 'gE' in STM.variables:
        title('gE')
        for i in observation_list:
            plot(STM.t/ms, STM.gE[i])
    if len(observation_list) < 5:
        legend(observation_list, fontsize=11)

    subplot(2,2,4)
    if 'gi' in STM.variables:
        title('gi')
        for i in observation_list:
            plot(STM.t/ms, STM.gi[i])
    elif 'gI' in STM.variables:
        title('gI')
        for i in observation_list:
            plot(STM.t/ms, STM.gI[i])
    if len(observation_list) < 5:
        legend(observation_list, fontsize=11)

    if savefig_:
        savefig(savefig_)
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


def plot_heading(h, headings, SPM_HEADING, T_outbound, time_step, N=8, figsize=(10,5), savefig_=None):
    figure(figsize=figsize)
    bins = numpy.linspace(-np.pi, np.pi, N+1)
    plt.plot(h)
    for b in bins:
        plt.axhline(b, ls='dashed', color='r', alpha=0.5)
    ylabel('angle[-pi, pi]')
    xlabel('steps')
    plt.savefig('plots/path.png')
    plt.show()

    figure(figsize=(10,5))
    plot(SPM_HEADING.t/ms, SPM_HEADING.i, '.k', markersize=6)
    plot(np.array(range(0, T_outbound*time_step, time_step)), np.argmax(headings,axis=1),'xr', markersize=3, alpha=0.2)
    plot(np.array(range(0, T_outbound*time_step, time_step)), N+np.argmax(headings,axis=1),'xr', markersize=3, alpha=0.2)
    ylabel('neuron index')
    xlabel('ms')
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_flow(flow, SPM_FLOW, figsize=(10,5), savefig_=None):
    figure(figsize=figsize)
    plt.plot(flow[:,0], label='L')
    plt.plot(flow[:,1], label='R')
    plt.legend()
    plt.savefig('plots/flow.png')
    plt.show()

    figure(figsize=(10,5))
    plot(SPM_FLOW.t/ms, SPM_FLOW.i, '.k', markersize=6)
    yticks([0,1],['L','R'])
    if savefig_:
        plt.savefig(savefig_)
    show()


def plot_rate_cx_log(matrix, max_rate, figsize=(10,5), savefig_=None):
    figure(figsize=figsize)
    plt.pcolormesh(max_rate*matrix, vmin=0, vmax=max_rate,
                   cmap='viridis', rasterized=True)    
    plt.colorbar()
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_rate_cx_log_spikes(matrix, max_rate, monitor, time_step, min_rate=0, title=None, figsize=(10,5), savefig_=None):
    plt.figure(figsize=figsize)
    plt.pcolormesh(max_rate*matrix, vmin=min_rate, vmax=max_rate*matrix.max(),
                   cmap='viridis', rasterized=True)

    plt.plot(monitor.t/ms / time_step, monitor.i+0.5, '.r')
    plt.colorbar()
    if title:
        plt.title(title)
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_motors_cx_log_spikes(matrix, max_rate, monitor, time_step, min_rate=0, title=None, figsize=(10,5), savefig_=None):
    plt.figure(figsize=figsize)
    plt.pcolormesh(matrix, vmin=min_rate, vmax=max_rate,
                   cmap='viridis', rasterized=True)

    plt.plot(monitor.t/ms / time_step, monitor.i+0.5, '.r')
    plt.colorbar()
    if title:
        plt.title(title)
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_gamma_factors(gamma_factors, tau_s, w_s,  
                       title='', xlabel='wE (nS)', ylabel='tauE (ms)', 
                       figsize=(11,7), savefig_=None):
    c = np.argwhere(gamma_factors == np.min(gamma_factors))[0]
    
    plt.figure(figsize=figsize)
    plt.pcolormesh(gamma_factors,cmap='viridis', rasterized=True)
    plt.plot(c[1]+0.5, c[0]+0.5, 'rx')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(len(tau_s))+0.5, tau_s)
    plt.xticks(np.arange(len(w_s))+0.5, w_s)
    plt.colorbar()
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_raster_plot(spm, title=None, figsize=(15,5), savefig_=None):
    plt.figure(figsize=figsize)
    if title:
        plt.title(title)
    brian_plot(spm)
    if savefig_:
        plt.savefig(savefig_)
    plt.show()

