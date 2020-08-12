import numpy as np
from brian2 import ms, Hz, Network, PoissonGroup, NeuronGroup, Synapses, NetworkOperation
import scipy

import cx_spiking.network_creation as nc
import cx_spiking.inputs

from cx_spiking.constants import *
# Number of cells

class CX_SPIKING(object):
    '''
    Class for a Spiking Central Complex object
    '''

    def __init__(self, 
                 neuron_eqs,
                 threshold_eqs,
                 reset_eqs, 
                 headings,
                 flow,
                 cpu4_method,
                 mem_gain_outbound,
                 decay_outbound,
                 mem_gain_inbound,
                 decay_inbound,
                 acceleration=0.3, 
                 drag=0.15,
                 rotation_factor=0.01,
                 time_step=20,
                 T_outbound=1500,
                 T_inbound=1500):

        ######################################
        ### PARAMETERS
        ######################################
        self.T_outbound = T_outbound
        self.T_inbound = T_inbound
        self.T = T_outbound + T_inbound

        self.time_step = time_step

        self.mem_gain_outbound = mem_gain_outbound
        self.mem_gain_inbound = mem_gain_inbound
        self.decay_outbound = decay_outbound
        self.decay_inbound = decay_inbound

        self.rotation_factor = rotation_factor

        self.cpu4_method = cpu4_method

        self.acceleration = acceleration
        self.drag = drag

        self.populations_size()

        ######################################
        ### INPUTS 
        ######################################
        self.inputs = self.construct_inputs(headings, flow)
        self.inputs_spike_monitors = self.create_inputs_spike_monitors()


        ######################################
        ### NETWORK 
        ######################################
        self.connectivity_matrices()
        
        self.neural_populations = self.construct_neural_populations(neuron_eqs, threshold_eqs, reset_eqs)

        self.populations_spike_monitors = self.create_populations_spike_monitors()

        self.synapses = self.construct_synapses()

        ######################################
        ### NETWORK OPERATIONS
        ######################################
        self.network_operations_utilities()
        self.network_operations = self.construct_network_operations(self.cpu4_method)

        ######################################
        ### PLOTTING
        ######################################
        self.bee_coords = np.zeros((self.T, 2))

        self.update_bee_position_net_op = NetworkOperation(self.update_bee_position, dt=self.time_step*ms, 
                                                           when='end', order=4, name='update_bee_position')
        self.network_operations.append(self.update_bee_position_net_op)

        net = Network()
        net.add(self.inputs,
                self.inputs_spike_monitors,
                self.neural_populations,
                self.populations_spike_monitors,
                self.synapses,
                self.network_operations)
        self.net = net
        self.net.store('initialised')

    def run_outbound(self, steps=-1, store_name='outbound'):
        self.net.restore('initialised')
        self.net['update_inputs'].active = False
        self.net['CPU4_accumulator_inbound'].active = False

        if steps < 0  or steps > self.T_outbound:
            steps = self.T_outbound

        print(f'Run network for {steps} steps')
      
        self.net.run(steps * self.time_step * ms, report='text')

        self.net.store(store_name)


    def run_inbound(self, steps=-1, restore_name='outbound', store_name=None):
        self.net.restore(restore_name)

        self.net['update_inputs'].active = True
        self.net['CPU4_accumulator'].active = False
        self.net['CPU4_accumulator_inbound'].active = True

        if steps < 0 or steps > self.T_inbound:
            steps = self.T_inbound

        print(f'Run network for {steps} steps')

        self.net.run(steps * self.time_step * ms, report='text')

        if store_name:
            self.net.store(store_name)


    def run_all(self):
        self.run_outbound(steps=self.T_outbound, store_name='outbound')
        self.run_inbound(steps=self.T_inbound, restore_name='outbound')
        pass


    def construct_inputs(self, headings, flow):
        self.headings_hz = headings*Hz
        self.P_HEADING = PoissonGroup(self.N_TL2, rates=self.headings_hz[0,:], name='P_HEADING')

        self.flow_hz = flow*Hz
        self.P_FLOW = PoissonGroup(self.N_TN2, rates=self.flow_hz[0,:], name='P_FLOW')

        #global CPU4_memory_stimulus
        self.CPU4_memory_stimulus = CPU_MEMORY_starting_value * np.ones((self.T, self.N_CPU4)) * Hz
        self.P_CPU4_MEMORY = PoissonGroup(self.N_CPU4, rates=self.CPU4_memory_stimulus[0,:], name='P_CPU4_MEMORY')

        return [self.P_HEADING, self.P_FLOW, self.P_CPU4_MEMORY]


    def construct_neural_populations(self, neuron_eqs, threshold_eqs, reset_eqs):
        self.G_TL2 = nc.generate_neuron_groups(self.N_TL2, neuron_eqs, threshold_eqs, reset_eqs, 
                                               TL2_neuron_params, name='TL2')
        self.G_CL1 = nc.generate_neuron_groups(self.N_CL1, neuron_eqs, threshold_eqs, reset_eqs, 
                                               CL1_neuron_params, name='CL1')
        self.G_TB1 = nc.generate_neuron_groups(self.N_TB1, neuron_eqs, threshold_eqs, reset_eqs, 
                                               TB1_neuron_params, name='TB1')
        self.G_TN2 = nc.generate_neuron_groups(self.N_TN2, neuron_eqs, threshold_eqs, reset_eqs, 
                                               TN2_neuron_params, name='TN2')
        self.G_CPU4 = nc.generate_neuron_groups(self.N_CPU4, neuron_eqs, threshold_eqs, reset_eqs, 
                                                CPU4_neuron_params, name='CPU4')
        self.G_CPU1A = nc.generate_neuron_groups(self.N_CPU1A, neuron_eqs, threshold_eqs, reset_eqs, 
                                                 neuron_params, name='CPU1A')
        self.G_CPU1B = nc.generate_neuron_groups(self.N_CPU1B, neuron_eqs, threshold_eqs, reset_eqs, 
                                                 neuron_params, name='CPU1B')
        self.G_PONTINE = nc.generate_neuron_groups(self.N_PONTINE, neuron_eqs, threshold_eqs, reset_eqs, 
                                                   neuron_params, name='PONTINE')
        self.G_MOTOR = nc.generate_neuron_groups(self.N_MOTOR, neuron_eqs, threshold_eqs, reset_eqs, 
                                                 neuron_params, name='MOTOR')

        return [self.G_TL2, self.G_CL1, self.G_TB1, self.G_TN2, self.G_CPU4,
                self.G_CPU1A, self.G_CPU1B, self.G_PONTINE, self.G_MOTOR]


    def construct_synapses(self):
        # Inputs
        self.S_P_HEADING_TL2 = nc.connect_synapses(self.P_HEADING, self.G_TL2, self.W_HEADING_TL2, model=synapses_model, 
                                                   params=H_TL2_synapses_params, on_pre=synapses_eqs_ex, name='S_P_HEADING_TL2')
        self.S_P_FLOW_TN2 = nc.connect_synapses(self.P_FLOW, self.G_TN2, self.W_FLOW_TN2, model=synapses_model, 
                                                params=F_TN2_synapses_params, on_pre=synapses_eqs_ex, name='S_P_FLOW_TN2')
        # TL2
        self.S_TL2_CL1 = nc.connect_synapses(self.G_TL2, self.G_CL1, self.W_TL2_CL1, model=synapses_model, 
                                             params=TL2_CL1_synapses_params, on_pre=synapses_eqs_ex, name='S_TL2_CL1')
        # CL1
        self.S_CL1_TB1 = nc.connect_synapses(self.G_CL1, self.G_TB1, self.W_CL1_TB1, model=synapses_model, 
                                             params=CL1_TB1_synapses_params, on_pre=synapses_eqs_ex, name='S_CL1_TB1')
        # TN2
        self.S_TN2_CPU4 = nc.connect_synapses(self.G_TN2, self.G_CPU4, self.W_TN2_CPU4, model=synapses_model, 
                                              params=TN2_CPU4_synapses_params, on_pre=synapses_eqs_ex, name='S_TN2_CPU4')
        # TB1
        self.S_TB1_TB1 = nc.connect_synapses(self.G_TB1, self.G_TB1, self.W_TB1_TB1, model=synapses_model, 
                                             params=TB1_TB1_synapses_params, on_pre=synapses_eqs_in, name='S_TB1_TB1')
        self.S_TB1_CPU4 = nc.connect_synapses(self.G_TB1, self.G_CPU4, self.W_TB1_CPU4, model=synapses_model, 
                                              params=TB1_CPU4_synapses_params, on_pre=synapses_eqs_in, name='S_TB1_CPU4')
        self.S_TB1_CPU1A = nc.connect_synapses(self.G_TB1, self.G_CPU1A, self.W_TB1_CPU1A, model=synapses_model, 
                                               params=synapses_params, on_pre=synapses_eqs_in, name='S_TB1_CPU1A')
        self.S_TB1_CPU1B = nc.connect_synapses(self.G_TB1, self.G_CPU1B, self.W_TB1_CPU1B, model=synapses_model, 
                                               params=synapses_params,  on_pre=synapses_eqs_in, name='S_TB1_CPU1B')
        # CPU4 accumulator
        self.S_CPU4_M_PONTINE = nc.connect_synapses(self.P_CPU4_MEMORY, self.G_PONTINE, self.W_CPU4_PONTINE, model=synapses_model, 
                                                    params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_PONTINE')
        self.S_CPU4_M_CPU1A = nc.connect_synapses(self.P_CPU4_MEMORY, self.G_CPU1A, self.W_CPU4_CPU1A, model=synapses_model, 
                                                  params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_CPU1A')
        self.S_CPU4_M_CPU1B = nc.connect_synapses(self.P_CPU4_MEMORY, self.G_CPU1B, self.W_CPU4_CPU1B, model=synapses_model, 
                                                  params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU4_M_CPU1B')
        # Pontine
        self.S_PONTINE_CPU1A = nc.connect_synapses(self.G_PONTINE, self.G_CPU1A, self.W_PONTINE_CPU1A, model=synapses_model, 
                                                   params=synapses_params, on_pre=synapses_eqs_in, name='S_PONTINE_CPU1A')
        self.S_PONTINE_CPU1B = nc.connect_synapses(self.G_PONTINE, self.G_CPU1B, self.W_PONTINE_CPU1B, model=synapses_model, 
                                                   params=synapses_params, on_pre=synapses_eqs_in, name='S_PONTINE_CPU1B')
        # CPU1A
        self.S_CPU1A_MOTOR = nc.connect_synapses(self.G_CPU1A, self.G_MOTOR, self.W_CPU1A_MOTOR, model=synapses_model, 
                                                 params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU1A_MOTOR')
        # CPU1B
        self.S_CPU1B_MOTOR = nc.connect_synapses(self.G_CPU1B, self.G_MOTOR, self.W_CPU1B_MOTOR, model=synapses_model, 
                                                 params=synapses_params, on_pre=synapses_eqs_ex, name='S_CPU1B_MOTOR')

        return [self.S_P_HEADING_TL2, self.S_P_FLOW_TN2, self.S_TL2_CL1, self.S_CL1_TB1, self.S_TN2_CPU4,
                self.S_TB1_TB1, self.S_TB1_CPU4, self.S_TB1_CPU1A, self.S_TB1_CPU1B, 
                self.S_CPU4_M_PONTINE, self.S_CPU4_M_CPU1A, self.S_PONTINE_CPU1A, self.S_PONTINE_CPU1B,
                self.S_CPU1A_MOTOR, self.S_CPU1B_MOTOR]
    

    def create_populations_spike_monitors(self):
        self.SPM_TL2 = SpikeMonitor(self.G_TL2, name='SPM_TL2')
        self.SPM_CL1 = SpikeMonitor(self.G_CL1, name='SPM_CL1')
        self.SPM_TB1 = SpikeMonitor(self.G_TB1, name='SPM_TB1')
        self.SPM_TN2 = SpikeMonitor(self.G_TN2, name='SPM_TN2')
        self.SPM_CPU4 = SpikeMonitor(self.G_CPU4, name='SPM_CPU4')
        self.SPM_CPU1A = SpikeMonitor(self.G_CPU1A, name='SPM_CPU1A')
        self.SPM_CPU1B = SpikeMonitor(self.G_CPU1B, name='SPM_CPU1B')
        self.SPM_PONTINE = SpikeMonitor(self.G_PONTINE, name='SPM_PONTINE')
        self.SPM_MOTOR = SpikeMonitor(self.G_MOTOR, name='SPM_MOTOR')

        return [self.SPM_TL2, self.SPM_CL1, self.SPM_TB1, self.SPM_TN2, self.SPM_CPU4,
                self.SPM_CPU1A, self.SPM_CPU1B, self.SPM_PONTINE, self.SPM_MOTOR]


    def create_inputs_spike_monitors(self):
        self.SPM_HEADING = SpikeMonitor(self.P_HEADING, name='SPM_HEADING')
        self.SPM_FLOW = SpikeMonitor(self.P_FLOW, name='SPM_FLOW')
        self.SPM_CPU4_MEMORY = SpikeMonitor(self.P_CPU4_MEMORY, name='SPM_P_CPU4_MEMORY')

        return [self.SPM_HEADING, self.SPM_FLOW, self.SPM_CPU4_MEMORY]


    def populations_size(self):
        self.N_TL2 = 16
        self.N_CL1 = 16
        self.N_TB1 = 8
        self.N_TN2 = 2
        self.N_CPU4 = 16
        self.N_CPU1A = 14
        self.N_CPU1B = 2
        self.N_CPU1 =  self.N_CPU1A + self.N_CPU1B
        self.N_PONTINE = 16

        self.N_MOTOR = 2


    def connectivity_matrices(self, weight=1.):
        self.W_HEADING_TL2 = np.eye(self.N_TL2)
        self.W_FLOW_TN2 = np.eye(self.N_TN2)
        # Act as if CL1 cells were inverting TL2 output
        # by shifting it by 180 degrees 
        self.W_TL2_CL1 = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
        ])
        self.W_CL1_TB1 = np.tile(np.eye(self.N_TB1), 2)
        self.W_TB1_TB1 = self.generate_TB1_TB1_weights(weight=weight)
        self.W_TB1_CPU1A = np.tile(np.eye(self.N_TB1), (2, 1))[1:self.N_CPU1A+1, :]
        self.W_TB1_CPU1B = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 0]])
        self.W_TB1_CPU4 = np.tile(np.eye(self.N_TB1), (2, 1))
        self.W_TN2_CPU4 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).T
        self.W_CPU4_CPU1A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        self.W_CPU4_CPU1B = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
        ])
        self.W_PONTINE_CPU1A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15
        ])
        self.W_PONTINE_CPU1B = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #9
        ])
        self.W_CPU4_PONTINE = np.eye(self.N_CPU4)

        # Not sure about these
        self.W_CPU1A_MOTOR = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        ])
        self.W_CPU1B_MOTOR = np.array([[0, 1],
                                [1, 0]])


    def generate_TB1_TB1_weights(self, weight=1.):
        """
        Weight matrix to map inhibitory connections from TB1 to other neurons
        
        from Thomas Stone's path-integration repo
        """
        W = np.zeros([self.N_TB1, self.N_TB1])
        sinusoid = -(np.cos(np.linspace(0, 2*np.pi, self.N_TB1, endpoint=False)) - 1)/2
        for i in range(self.N_TB1):
            values = np.roll(sinusoid, i)
            W[i, :] = values
        return weight * W


    def get_agent_timestep(self, t, sim_time_step):
        # 0.5 added for numerical stability, python integer rounds down
        # sometimes t/ms = x.99999 and it gets rounded to x-1
        return int((t/ms + 0.5) / sim_time_step)


    def make_angle(self, theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi


    def compute_peak(self, neurons_responses, ref_angles):
        # trick to get the correct weighted average of where the heading is
        # create a list with all the angles between [-pi,pi] repeated by their count
        # so [0,2,0,0,1,0,0,1] will be [-1.963, -1.963, 0.392, 2.748] and then compute
        # circular mean between [-pi, pi]
        tmp = [angle for i, angle in enumerate(ref_angles) for neuron in range(neurons_responses[i])]
        # -pi/8 because we center the neurons at the center of their pi/4 receptive fields
        peak = scipy.stats.circmean(tmp, low=-np.pi, high=np.pi) - np.pi/8
        return self.make_angle(peak)


    def get_next_velocity(self, heading, velocity, acceleration=0.3, drag=0.15):
        def thrust(theta, acceleration):
            return np.array([np.sin(theta), np.cos(theta)]) * acceleration
        v = velocity + thrust(heading, acceleration).flatten()
        v -= drag * v
        return np.clip(v, 0, np.inf)


    def network_operations_utilities(self):
        # Record headings
        self.ref_angles = np.linspace(-np.pi+np.pi/8, np.pi+np.pi/8, self.N_TB1, endpoint=False)
        self.heading_angles = np.zeros(self.T)

        # Record velocities
        self.max_velocity = 12 
        self.velocities = np.zeros((self.T, self.N_TN2))

        # CPU4 accumulation
        self.CPU4_memory_history = CPU_MEMORY_starting_value * np.ones((self.T, self.N_CPU4))
        self.CPU4_memory = CPU_MEMORY_starting_value * np.ones(self.N_CPU4)

        # Update inputs
        self.new_heading_dir = np.zeros(self.T)
        self.new_velocities = np.zeros((self.T, self.N_TN2))
        self.rotations = np.zeros(self.T)


    def construct_network_operations(self, cpu4_method):
        # Set up network operations
        self.extract_heading_net_op = NetworkOperation(self.extract_heading, dt=self.time_step*ms, 
                                                    when='start', order=0, name='extract_heading')
        self.extract_velocity_net_op = NetworkOperation(self.extract_velocity, dt=self.time_step*ms, 
                                                        when='start', order=1, name='extract_velocity')

        if cpu4_method == 1:
            self.CPU4_accumulator_net_op = NetworkOperation(self.CPU4_accumulator_method1, dt=self.time_step*ms, 
                                                            when='start', order=2, name='CPU4_accumulator')
            self.CPU4_accumulator_inbound_net_op = NetworkOperation(self.CPU4_accumulator_inbound_method1, dt=self.time_step*ms, 
                                                                    when='start', order=2, name='CPU4_accumulator_inbound')


        self.update_inputs_net_op = NetworkOperation(self.update_inputs, dt=self.time_step*ms, 
                                                     when='start', order=3, name='update_inputs')


        self.set_rates = NetworkOperation(self.set_rates, dt=self.time_step*ms, 
                                          when='start', order=4, name='set_rates')


        return [self.extract_heading_net_op, self.extract_velocity_net_op, 
                self.CPU4_accumulator_net_op, self.CPU4_accumulator_inbound_net_op, 
                self.update_inputs_net_op, self.set_rates]


    def extract_heading(self, t):
        global ref_angles, heading_angles

        timestep = self.get_agent_timestep(t, self.time_step)
        
        if t < self.time_step*ms:
            self.heading_angles[timestep] = self.compute_peak([0,0,0,1,1,0,0,0], self.ref_angles)
            self.G_TB1.spike_count = 0
            return

        neurons_responses = self.G_TB1.spike_count

        # if no responses copy the previous heading
        if np.sum(neurons_responses) > 0:
            self.heading_angles[timestep] = self.compute_peak(neurons_responses, self.ref_angles)
        else:
            self.heading_angles[timestep] = self.heading_angles[timestep-1]
        
        self.G_TB1.spike_count = 0


    def extract_velocity(self, t):
        global velocities, max_velocity
        
        timestep = self.get_agent_timestep(t, self.time_step)

        if t < self.time_step*ms:
            self.velocities[timestep] = [0,0]
            self.G_TN2.spike_count = 0
            return
        neurons_responses = self.G_TN2.spike_count

        neurons_responses = np.clip(neurons_responses, 0, self.max_velocity)
        self.velocities[timestep] = neurons_responses / self.max_velocity

        self.G_TN2.spike_count = 0


    def CPU4_accumulator_method1(self, t):
        timestep = self.get_agent_timestep(t, self.time_step)
        
        if t < self.time_step*ms:
            return

        neurons_responses = self.G_CPU4.spike_count

        mem_update = neurons_responses 
        self.CPU4_memory = self.CPU4_memory_history[timestep-1,:]
        self.CPU4_memory += mem_update * self.mem_gain_outbound
        self.CPU4_memory -= self.decay_outbound * (1./(mem_update+0.1))
        self.CPU4_memory = np.clip(self.CPU4_memory, 0, np.inf)
        self.CPU4_memory_history[timestep,:] = self.CPU4_memory

        self.CPU4_memory_stimulus[timestep,:] = self.CPU4_memory * Hz
        
        self.G_CPU4.spike_count = 0


    def CPU4_accumulator_inbound_method1(self, t):
        global CPU4_memory, CPU4_memory_history, CPU4_memory_stimulus
        
        timestep = self.get_agent_timestep(t, self.time_step)
        
        if t < self.time_step*ms:
            return

        neurons_responses = self.G_CPU4.spike_count

        mem_update = neurons_responses 
        self.CPU4_memory = self.CPU4_memory_history[timestep-1,:]
        self.CPU4_memory += mem_update * self.mem_gain_inbound
        self.CPU4_memory -= self.decay_inbound #* (1./(mem_update+0.1))
        self.CPU4_memory = np.clip(self.CPU4_memory, 0, np.inf)
        self.CPU4_memory_history[timestep,:] = self.CPU4_memory

        self.CPU4_memory_stimulus[timestep,:] = self.CPU4_memory * Hz
        
        self.G_CPU4.spike_count = 0


    def update_inputs(self, t):
        timestep = self.get_agent_timestep(t, self.time_step)

        #### motor response - rotation
        motor_responses = self.G_MOTOR.spike_count
        rotation = np.sign(motor_responses[0] - motor_responses[1])
        self.G_MOTOR.spike_count = 0
        
        #### heading
        # previous heading
        prev_heading = np.array([self.heading_angles[timestep]])
        # compute spikes based on old heading and rotation using fixed angle "step" of 22.5 degrees 
        new_heading = self.make_angle(prev_heading + self.rotation_factor) # mean and median rotation found from rate model
        self.new_heading_dir[timestep] = new_heading
        self.rotations[timestep] = rotation * self.rotation_factor
        new_headings = cx_spiking.inputs.compute_headings(new_heading, N=self.N_TL2//2, vmin=5, vmax=100)
        new_headings = np.tile(new_headings, 2) 
        # save new heading
        self.headings_hz[timestep,:] = new_headings * Hz

        #### velocity
        velocity = np.array(self.velocities[timestep,:])
        updated_v = self.get_next_velocity(new_heading, velocity, rotation, 
                                           self.acceleration, self.drag)
        self.new_velocities[timestep,:] = updated_v
        new_flow = cx_spiking.inputs.compute_flow(new_heading, updated_v, baseline=50, 
                                                vmin=0, vmax=50, inbound=True)
        self.flow_hz[timestep,:] = new_flow * Hz


    def set_rates(self, t):
        timestep = self.get_agent_timestep(t, self.time_step)

        if t < self.time_step*ms:
            return
        self.P_HEADING.rates = self.headings_hz[timestep,:]
        self.P_FLOW.rates = self.flow_hz[timestep,:]
        self.P_CPU4_MEMORY.rates = self.CPU4_memory_stimulus[timestep,:]


    def update_bee_position(self, t):
        timestep = self.get_agent_timestep(t, self.time_step)

        if t < self.time_step*ms:
            return

        # Compute speed component
        speed = 1 + np.clip(np.linalg.norm(self.velocities[timestep]), 0, 1) 

        angle = self.heading_angles[timestep]

        # x should be cos and y should be sin 
        # keep compatibility with stone's code (plotter.py : line 79)
        x_comp = np.sin(angle) #* speed
        y_comp = np.cos(angle) #* speed

        bee_x = self.bee_coords[timestep-1,0] + x_comp
        bee_y = self.bee_coords[timestep-1,1] + y_comp
        self.bee_coords[timestep,0] = bee_x 
        self.bee_coords[timestep,1] = bee_y 


    def plot_bee_position(self):
        import matplotlib.pyplot as plt
        f = plt.figure(figsize=(8,8))
        plt.text(0, 0, 'N', fontsize=12, fontweight='heavy', color='k', ha='center', va='center')
        plt.plot(self.bee_coords[:self.T_outbound,0], self.bee_coords[:self.T_outbound,1], color='purple', lw=1, label='Outbound')
        plt.plot(self.bee_coords[self.T_outbound:,0], self.bee_coords[self.T_outbound:,1], color='green', lw=1, label='Return')
        plt.legend()
        plt.axis('scaled')
        plt.show()
