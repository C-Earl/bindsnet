import numpy as np
import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight


# Class for a spiking neural network that uses STDP and Q-Learning to learn
class STDP_Q_Learning(Network):
  def __init__(self,
               in_size: int,  # Number of association (input) neurons
               out_size: int,  # Number of motor control neurons
               w_exc_out: np.ndarray,  # Association to motor control weights
               w_out_out: np.ndarray,  # Motor control to motor control weights
               alpha: float,  # Q-Learning Learning rate (Q-table learning)
               gamma: float,  # Q-Learning discount factor
               num_actions: int,  # Number of possible actions
               wmin: float,  # Minimum synaptic weight value
               wmax: float,  # Maximum synaptic weight value
               decay: float,  # Weight decay factor
               lr: float,  # Learning rate for STDP (synaptic learning)
               hyper_params: dict,  # Dictionary of hyperparameters
               device: str = 'cpu'):
    super().__init__()

    # Motor population size should be multiple of number of actions
    assert out_size % num_actions == 0, "Number of motor control neurons must be multiple of number of actions"
    self.motor_pop_size = int(out_size / num_actions)


    ## Layers ##
    input = Input(n=in_size)    # Inputs from association area
    output = AdaptiveLIFNodes(  # Motor control neurons
      n=out_size,
      thresh=hyper_params['thresh_out'],
      theta_plus=hyper_params['theta_plus_out'],
      refrac=hyper_params['refrac_out'],
      reset=hyper_params['reset_out'],
      tc_theta_decay=hyper_params['tc_theta_decay_out'],
      tc_decay=hyper_params['tc_decay_out'],
      traces=True,
    )
    output_monitor = Monitor(output, ["s"], device=device)
    input_monitor = Monitor(input, ["s"], device=device)
    self.output_monitor = output_monitor
    self.input_monitor = input_monitor
    self.add_monitor(input_monitor, name='input_monitor')
    self.add_monitor(output_monitor, name='output_monitor')
    self.add_layer(input, name='input')
    self.add_layer(output, name='output')


    ## Connections ##
    in_out_wfeat = Weight(name='in_out_weight_feature', value=torch.Tensor(w_exc_out))
    in_out_conn = MulticompartmentConnection(
      source=input, target=output,
      device=device, pipeline=[in_out_wfeat],
    )
    out_out_wfeat = Weight(name='out_out_weight_feature', value=torch.Tensor(w_out_out))
    out_out_conn = MulticompartmentConnection(
      source=output, target=output,
      device=device, pipeline=[out_out_wfeat],
    )
    self.add_connection(in_out_conn, source='input', target='output')
    self.add_connection(out_out_conn, source='output', target='output')
    self.weights = in_out_wfeat
    self.w_mask = in_out_wfeat.value != 0

    ## Q-Learning Parameters ##
    self.gamma = gamma
    self.alpha = alpha
    self.num_actions = num_actions
    self.q_table = {}

    ## STDP Parameters ##
    self.wmin, self.wmax = wmin, wmax
    self.lr = lr
    self.synapse_change_monitor = torch.zeros(())

  def STDP_RL(self, reward: float, input_spikes, output_spikes):
    # Calculate STDP learning eligibility
    eligibility = torch.outer(input_spikes.sum(0), output_spikes.sum(0))

    # Normalize eligibility so each presynaptic neuron (row) sums to 0
    # Finite resource limitation
    eligibility = eligibility - torch.mean(eligibility, dim=1, keepdim=True)

    # Add noise to reward
    # Exploration when reward is 0
    reward += np.random.normal(0, 0.1)

    # Update weights according to reward and eligibility
    dw = self.lr * eligibility * reward
    self.weights.value += dw
    self.weights.value = torch.clamp(self.weights.value, self.wmin, self.wmax)

  def Q_Learning(self, state: tuple, action: int, reward: float, next_state: tuple):
    if state not in self.q_table:
      self.q_table[state] = np.zeros(self.num_actions)
    if next_state not in self.q_table:
      self.q_table[next_state] = np.zeros(self.num_actions)
    org_val = self.q_table[state][action]
    next_max_val = self.q_table[next_state].max()
    self.q_table[state][action] = org_val + self.alpha * (reward + self.gamma * next_max_val - org_val)
    delta_q = self.q_table[state][action] - org_val
    return delta_q

  # Take in_spikes (association area spikes) and return action based on motor area spikes
  def select_action(self, in_spikes: np.ndarray, sim_time: int):
    self.run(inputs={"input": torch.Tensor(in_spikes)}, time=sim_time)
    out_spikes = self.output_monitor.get("s")
    out_spikes = out_spikes.squeeze(1)    # Remove batch dimension
    # If no spikes, return random action
    # Artificial spikes to encourage STDP learning
    if torch.max(out_spikes) == 0:
      action = np.random.randint(self.num_actions)
      out_spikes = torch.zeros_like(out_spikes)
      motor_pop_range = (action * self.motor_pop_size, (action + 1) * self.motor_pop_size)
      out_spikes[:, motor_pop_range[0]:motor_pop_range[1]] = torch.rand(sim_time, self.motor_pop_size) < 0.05
    else:
      summed_spikes = out_spikes.sum(0)
      max_val = summed_spikes.numpy().max()
      max_inds = np.where(summed_spikes == max_val)[0]
      action = max_inds[np.random.randint(0, len(max_inds))]
      # action = torch.argmax(out_spikes.reshape(sim_time, self.num_actions).sum(0))
    return action, out_spikes

  def plot_weights(self, ax):
    w = self.weights.value
    ax.imshow(w, cmap='viridis')
    ax.set_title('Synaptic Weights')
    ax.set_xlabel('Motor Control Neurons')
    ax.set_ylabel('Association Neurons')
    ax.set_aspect('auto')
    return ax

  def plot_spikes(self, ax, spikes):
    ax.imshow(spikes.T, aspect='auto', cmap='binary')
    ax.set_title('Spikes')
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    return ax

