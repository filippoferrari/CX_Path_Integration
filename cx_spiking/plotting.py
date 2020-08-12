import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
from brian2tools import *

import cx_spiking.inputs
from cx_spiking.constants import *

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


def colorbar(mappable, title=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.03)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(title)
    plt.sca(last_axes)
    return cbar


def plot_rate_cx_log_spikes(matrix, max_rate, monitor, time_step, min_rate=0, 
                            title=None, colorbar_title='impulses/s', figsize=(10,5), savefig_=None, xlim=[]):
    plt.figure(figsize=figsize)
    p = plt.pcolormesh(max_rate*matrix, vmin=min_rate, vmax=max_rate*matrix.max(),
                   cmap='viridis', rasterized=True)

    plt.plot(monitor.t/ms / time_step, monitor.i+0.5, '.r')
    plt.yticks(np.arange(0,matrix.shape[0],2)+0.5, np.arange(0,matrix.shape[0], 2))
    colorbar(p, colorbar_title)
    if len(xlim):
        plt.xlim(xlim)
    if title:
        plt.title(title)
    if savefig_:
        plt.savefig(savefig_)
    plt.show()


def plot_motors_cx_log_spikes(matrix, max_rate, monitor, time_step, min_rate=0, 
                              title=None, colorbar_title='impulses/s', figsize=(10,5), savefig_=None, xlim=[]):
    plt.figure(figsize=figsize)
    p = plt.pcolormesh(matrix, vmin=min_rate, vmax=max_rate,
                   cmap='viridis', rasterized=True)

    plt.plot(monitor.t/ms / time_step, monitor.i+0.5, '.r')
    colorbar(p, colorbar_title)
    if len(xlim):
        plt.xlim(xlim)
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



def plot_inputs(spiking_cx, h, v):
    plt.figure(figsize=(15,6))
    plt.plot((h[:spiking_cx.T_outbound] * 4 / np.pi + 0.5) % 8 , '.r', markersize=4)
    plt.plot((h[:spiking_cx.T_outbound] * 4 / np.pi + 0.5) % 8 + 8 , '.r', markersize=4)

    plt.axhline(8, color='k')

    p = plt.pcolormesh(spiking_cx.headings_hz[:spiking_cx.T_outbound,:].T/Hz, rasterized=True)

    plt.xlabel('Simulation step')
    plt.ylabel('Neuron index')
    plt.yticks(np.arange(0, spiking_cx.N_TL2) + 0.5, np.arange(1, spiking_cx.N_TL2+1))
    colorbar(p, 'impulses/s')
    plt.title('Heading conversion to spikes using von Mises')
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(spiking_cx.flow_hz[:spiking_cx.T_outbound,0]/Hz /100, 'r')
    plt.plot(spiking_cx.flow_hz[:spiking_cx.T_outbound,1]/Hz /100+1, 'r')
    plt.axhline(1, color='k')
    plt.xlabel('Simulation step')
    plt.ylabel('Neuron index')
    plt.yticks(np.arange(0, spiking_cx.N_TN2) + 0.5, np.arange(1, spiking_cx.N_TN2+1))
    p = plt.pcolormesh(spiking_cx.flow_hz[:spiking_cx.T_outbound,:].T/Hz)
    colorbar(p, 'impulses/s')
    plt.title('Flow conversion to spikes')
    plt.show()



def plot_populations_outbound(spiking_cx, cx_log, figsize=(15,5)):
    plot_populations(spiking_cx, cx_log, xlim=[0, spiking_cx.T_outbound], figsize=figsize)


def plot_populations_inbound(spiking_cx, cx_log, figsize=(15,5)):
    plot_populations(spiking_cx, cx_log, xlim=[spiking_cx.T_outbound, spiking_cx.T], figsize=figsize)


def plot_populations(spiking_cx, cx_log, xlim, figsize=(15,5)):
    plot_rate_cx_log_spikes(cx_log.tn2, TN2_spike_rates_max, spiking_cx.SPM_TN2, spiking_cx.time_step, 
                            title='TN2',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.tl2, TL2_spike_rates, spiking_cx.SPM_TL2, spiking_cx.time_step, 
                            title='TL2',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.cl1, CL1_spike_rates, spiking_cx.SPM_CL1, spiking_cx.time_step, 
                            title='CL1',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.tb1, TB1_spike_rates, spiking_cx.SPM_TB1, spiking_cx.time_step, 
                            title='TB1',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.cpu4, CPU4_spike_rates, spiking_cx.SPM_CPU4, spiking_cx.time_step, 
                            title='CPU4',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.memory, 1, spiking_cx.SPM_CPU4_MEMORY, spiking_cx.time_step, 
                            title='CPU4_memory',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.memory, 1, spiking_cx.SPM_PONTINE, spiking_cx.time_step, 
                            title='PONTINE over cpu4 memory',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.cpu1[1:-1,:], 1, spiking_cx.SPM_CPU1A, spiking_cx.time_step,
                            title='CPU1A',  figsize=figsize, xlim=xlim)

    plot_rate_cx_log_spikes(cx_log.cpu1[[0,-1],:], 1, spiking_cx.SPM_CPU1B, spiking_cx.time_step, 
                            title='CPU1B ',  figsize=figsize, xlim=xlim)

    motors = cx_spiking.inputs.compute_motors(cx_log.cpu1)
    plot_motors_cx_log_spikes(motors, motors.max(), spiking_cx.SPM_MOTOR, spiking_cx.time_step, 
                              min_rate=motors.min(), 
                              title='MOTOR',  figsize=figsize, xlim=xlim)


def plot_memory_outbound(spiking_cx, cx_log, figsize=(15,5)):
    plt.plot(spiking_cx.CPU4_memory_history[spiking_cx.T_outbound-1], label='code')
    plt.plot(cx_log.memory[:,spiking_cx.T_outbound-1] * spiking_cx.CPU4_memory_history[spiking_cx.T_outbound-1].max(), label='stone (rescaled)')
    plt.legend()
    plt.xlabel('neuron index')
    plt.ylabel('impulses/s')
    plt.title('CPU4 accumulation - end of outbound')
    plt.show()

    plt.figure(figsize=(15,5))
    ranges = range(spiking_cx.CPU4_memory_history.shape[1]//2)
    for r in ranges:
        plt.plot(spiking_cx.CPU4_memory_history[:spiking_cx.T_outbound,r], alpha=0.6, label=r)#, label=names[idx])
    plt.legend(bbox_to_anchor=(1.09, 1), title='Neuron index')
    #plt.xlim([1400,1500])
    plt.title('CPU4 accumulation L')
    plt.ylabel('impulses/s')
    plt.xlabel('Simulation steps')
    plt.show()

    plt.figure(figsize=figsize)
    ranges = range(spiking_cx.CPU4_memory_history.shape[1]//2)
    for r in ranges:
        plt.plot(spiking_cx.CPU4_memory_history[:spiking_cx.T_outbound,r+8], alpha=0.6, label=r+8)#, label=names[idx])
    plt.legend(bbox_to_anchor=(1.09, 1), title='Neuron index')
    #plt.xlim([1400,1500])
    plt.title('CPU4 accumulation R')
    plt.ylabel('impulses/s')
    plt.xlabel('Simulation steps')
    plt.show()


def plot_memory_inbound(spiking_cx, cx_log, figsize=(15,5)):
    plt.plot(spiking_cx.CPU4_memory_history[spiking_cx.T-1], label='code')
    plt.plot(cx_log.memory[:,spiking_cx.T-1] * spiking_cx.CPU4_memory_history[spiking_cx.T_outbound-1:spiking_cx.T-1].max(), label='stone (rescaled)')
    plt.legend()
    plt.xlabel('neuron index')
    plt.ylabel('impulses/s')
    plt.title('CPU4 accumulation - end of outbound')
    plt.show()

    plt.figure(figsize=(15,5))
    ranges = range(spiking_cx.CPU4_memory_history.shape[1]//2)
    for r in ranges:
        plt.plot(spiking_cx.CPU4_memory_history[spiking_cx.T_outbound:,r], alpha=0.6, label=r)#, label=names[idx])
    plt.legend(bbox_to_anchor=(1.09, 1), title='Neuron index')
    plt.title('CPU4 accumulation L')
    plt.ylabel('impulses/s')
    plt.xlabel('Simulation steps')
    plt.show()

    plt.figure(figsize=figsize)
    ranges = range(spiking_cx.CPU4_memory_history.shape[1]//2)
    for r in ranges:
        plt.plot(spiking_cx.CPU4_memory_history[spiking_cx.T_outbound:,r+8], alpha=0.6, label=r+8)#, label=names[idx])
    plt.legend(bbox_to_anchor=(1.09, 1), title='Neuron index')
    plt.title('CPU4 accumulation R')
    plt.ylabel('impulses/s')
    plt.xlabel('Simulation steps')
    plt.show()


# def plot_memory_accumulation(spiking_cx, xlim, title='CPU4 accumulation', figsize=(15,5)):

# def plot_summary(spiking_cx, cx_log, h, v):
#     plt.figure(figsize=(15,5))
#     plt.pcolormesh(spiking_cx.headings_hz.T/Hz)
#     plt.xlim([0,spiking_cx.T_outbound])
#     plt.title('spike headings - outbound')
#     plt.show()

#     plt.figure(figsize=(15,5))
#     brian_plot(spiking_cx.SPM_HEADING)
#     plt.xlim([0, spiking_cx.T_outbound * spiking_cx.time_step])
#     plt.title('SPM_HEADING - outbound')
#     plt.show()

#     plt.figure(figsize=(15,5))
#     brian_plot(spiking_cx.SPM_FLOW)
#     plt.xlim([0, spiking_cx.T_outbound* spiking_cx.time_step])
#     plt.title('SPM_FLOW - outbound')
#     plt.show()


#     plt.figure(figsize=(15,5))
#     plt.plot(spiking_cx.heading_angles, label='decoded')
#     plt.plot(h, label='real')
#     plt.xlim([0, spiking_cx.T_outbound])
#     plt.title('decoded angles - outbound')
#     plt.show()



#     TN2_spikes = cx_spiking.inputs.get_spikes_rates(spiking_cx.SPM_TN2, spiking_cx.N_TN2, spiking_cx.T_outbound, spiking_cx.time_step)
#     plt.figure(figsize=(15,5))
#     plt.title('TN2 spikes')
#     for idx, r in enumerate(TN2_spikes):
#         plt.plot(r, alpha=0.6, label=idx)
#     plt.legend()
#     plt.title('tn2 spikes')
#     plt.show()




#     plt.plot(spiking_cx.CPU4_memory_history[spiking_cx.T_outbound-1], label='code')
#     plt.plot(cx_log.memory[:,spiking_cx.T_outbound-1] * spiking_cx.CPU4_memory.max(), label='stone (rescaled)')
#     plt.legend()
#     plt.xlabel('neuron index')
#     plt.ylabel('impulses/s')
#     plt.title('CPU4 accumulation - end of outbound')
#     plt.show()




#     MOTOR_spikes =  inputs.get_spikes_rates(SPM_MOTOR, N_MOTOR, T_outbound, time_step)
#     plt.figure(figsize=(15,5))
#     plt.title('MOTOR')
#     for idx, r in enumerate(MOTOR_spikes):
#         plt.plot(r, alpha=0.6, label=idx)
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(15,5))
#     plt.plot((MOTOR_spikes[0,:]-MOTOR_spikes[1,:]))
#     plt.show()


#     plt.figure(figsize=(15,2))
#     #plt.plot(headings_diff)
#     plt.plot(np.sign(motors[0,:]-motors[1,:]), label='h')
#     plt.plot(np.sign((MOTOR_spikes[0,:]-MOTOR_spikes[1,:])), label='m')
#     plt.xlim([0,200])
#     plt.legend(bbox_to_anchor=(1.05, 1))
#     plt.show()