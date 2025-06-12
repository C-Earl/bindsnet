from collections import defaultdict

import numpy as np
import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight, Mask


# Class for a spiking neural network that uses STDP and Q-Learning to learn
class STDP_Q_Learning(Network):
  def __init__(self,
               in_size: int,  # Number of association (input) neurons
               out_size: int,  # Number of motor control neurons
               w_exc_out: np.ndarray,  # Association to motor control weights
               w_out_out: np.ndarray,  # Motor control to motor control weights
               exploration: float,  # Exploration rate for Q-Learning
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
    in_out_mask = Mask(name='in_out_mask', value=torch.Tensor(w_exc_out != 0).bool())
    in_out_conn = MulticompartmentConnection(
      source=input, target=output,
      device=device, pipeline=[in_out_wfeat, in_out_mask],
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
    self.exploration = exploration
    self.gamma = gamma
    self.alpha = alpha
    self.num_actions = num_actions
    self.q_table = {}

    ## STDP Parameters ##
    self.wmin, self.wmax = wmin, wmax
    self.lr = lr

  def STDP_RL(self, reward: float, input_spikes, output_spikes, action: int):
    # Ignore calculations if reward is zero
    if reward == 0:
      return
    else:
      # Ignore presynaptic neurons that are not active enough
      presynaptic_activity = input_spikes.sum(0)
      postsynaptic_activity = output_spikes.sum(0)
      presynaptic_activity[presynaptic_activity < 4] = 0

      # Calculate STDP learning eligibility
      eligibility = torch.outer(presynaptic_activity, postsynaptic_activity)

      # Normalize eligiblity
      # eligibility /= (eligibility.sum(1, keepdim=True) + 1e-8)

      # Set sum of presynaptic changes to 0 & apply mask
      # eligibility *= self.w_mask.float()
      # eligibility -= eligibility.sum(1, keepdim=True) / self.w_mask.sum(1, keepdim=True)

      # # Modify according to action taken
      # eligibility.reshape(-1, self.motor_pop_size, self.num_actions)


      # Add reward signal
      # eligibility *= reward

      # Prevent under/overflows
      # underflow_mask = (eligibility < 0) & (self.weights.value == self.wmin)
      # overflow_mask = (eligibility > 0) & (self.weights.value == self.wmax)
      # new_w = self.weights.value + eligibility
      # underflow_mask = (new_w < self.wmin).bool()
      # overflow_mask = (new_w > self.wmax).bool()
      # presynaptic_underflow_total = torch.where(underflow_mask, eligibility, torch.zeros_like(eligibility)).sum()
      # presynaptic_overflow_total = torch.where(overflow_mask, eligibility, torch.zeros_like(eligibility)).sum()
      # residual = presynaptic_overflow_total - presynaptic_underflow_total
      # if residual > 0:    # Redistribute extra resources
      #   candidate_synapses = (self.w_mask != 0) & (overflow_mask != 0)    # Only synapses not at max weight
      # elif residual < 0:    # Remove excess resources
      # eligibility[underflow_mask] = 0
      # eligibility[overflow_mask] = 0
      # underflow_modifier = presynaptic_underflow_total / (((self.w_mask != 0) & (underflow_mask != 0)).sum(1, keepdim=True) + 1e-8)
      # eligibility = torch.where((self.w_mask != 0) & (underflow_mask != 0), eligibility - underflow_modifier, eligibility)
      # # eligibility[(self.w_mask != 0) & (underflow_mask != 0)] = eligibility[(self.w_mask != 0) & (underflow_mask != 0)] - underflow_modifier
      # # eligibility *= self.w_mask.float()  # Apply weight mask
      # # eligibility[self.w_mask != 0] -= presynaptic_underflow_total / self.w_mask.sum(1, keepdim=True)
      # # eligibility[self.w_mask != 0] += presynaptic_overflow_total / self.w_mask.sum(1, keepdim=True)
      #
      # # Calculate for overflows
      # new_w = self.weights.value + (eligibility * reward)
      # new_w_clamped = torch.clamp(new_w, self.wmin, self.wmax)
      # delta = new_w_clamped - new_w


      ## Pre-synaptic resource limitation ##
      # Subtract row-wise mean eligibility for each row (pre-synaptic neuron)
      # presynaptic_eligibility = eligibility.sum(1, keepdim=True)
      # presynaptic_nonzeros = self.w_mask.sum(1, keepdim=True)
      # mean_eligibility = presynaptic_eligibility / presynaptic_nonzeros
      # eligibility -= mean_eligibility
      # eligibility[self.w_mask != 1] = 0  # Set masked weights to zero

      # Apply weight change

      eligibility = eligibility.reshape(presynaptic_activity.shape[0], self.num_actions, self.motor_pop_size)
      eligibility[:, action, :] *= reward
      eligibility[:, 0:action, :] *= -reward
      eligibility[:, action+1:, :] *= -reward
      eligibility = eligibility.reshape(presynaptic_activity.shape[0], self.motor_pop_size * self.num_actions)

      dw = self.lr * eligibility
      dw = dw * self.w_mask
      self.weights.value += dw



  def Q_Learning(self, state: tuple, action: int, reward: float, next_state: tuple):
    if state not in self.q_table:
      self.q_table[state] = np.zeros(self.num_actions)
    if next_state not in self.q_table:
      self.q_table[next_state] = np.zeros(self.num_actions)

    org_val = self.q_table[state][action]
    next_max_val = self.q_table[next_state].max()
    self.q_table[state][action] = (1-self.alpha)*org_val + self.alpha * (reward + self.gamma * next_max_val)

    state_max_val = self.q_table[state].max()
    if next_state != state:
      delta_q = next_max_val - state_max_val
    else:
      delta_q = -1
    return delta_q

  # Take in_spikes (association area spikes) and return action based on motor area spikes
  def select_action(self, in_spikes: np.ndarray, sim_time: int, explore=False):
    if explore:   # Choose random action
      action = np.random.randint(0, self.num_actions)
      out_spikes = torch.zeros((sim_time, self.num_actions, self.motor_pop_size))
      out_spikes[::100, action, :] = 1    # ~10hz spike rate per motor neuron
      out_spikes = out_spikes.reshape(sim_time, self.num_actions * self.motor_pop_size)
    else:
      self.run(inputs={"input": torch.Tensor(in_spikes)}, time=sim_time)
      out_spikes = self.output_monitor.get("s")
      out_spikes = out_spikes.squeeze(1)    # Remove batch dimension
      if torch.max(out_spikes) == 0:
        raise Exception("No spikes in output layer")
      else:
        summed_spikes = out_spikes.reshape(sim_time, self.num_actions, self.motor_pop_size).sum(2).sum(0)
        max_val = summed_spikes.numpy().max()
        max_inds = np.where(summed_spikes == max_val)[0]
        action = max_inds[np.random.randint(0, len(max_inds))]
    self.exploration *= .999999
    return action, out_spikes

  # Simulate the state and return the output spikes (nothing else)
  def simulate_state(self, in_spikes, sim_time):
    self.run(inputs={"input": torch.Tensor(in_spikes)}, time=sim_time)
    out_spikes = self.output_monitor.get("s")
    out_spikes = out_spikes.squeeze(1)  # Remove batch dimension
    return out_spikes

  def plot_weights(self, ax):
    w = self.weights.value
    ax.imshow(w, cmap='viridis', vmin=self.wmin, vmax=self.wmax)
    ax.set_title('Synaptic Weights')
    ax.set_xlabel('Motor Control Neurons')
    ax.set_ylabel('Association Neurons')
    ax.set_aspect('auto')
    return ax

  def plot_spikes(self, ax, spikes, title):
    ax.imshow(spikes.T, aspect='auto', cmap='spring')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    return ax

