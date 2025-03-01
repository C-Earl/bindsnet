import math
import random
import pickle as pkl
import numpy as np
import torch
from itertools import count

from matplotlib import pyplot as plt
# from triton.language import dtype

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDP
from scripts.Chris.DQN.Environment import Grid_Cell_Maze_Environment
from scripts.Chris.DQN.Memory import sparsify


class STDP_RL_Model(Network):
  def __init__(self, in_size, out_size, hyper_params, w_in_out, w_out_out, alpha, gamma,
               lr, decay, wmin, wmax, device='cpu'):
    super().__init__()

    ## Layers ##
    input = Input(n=in_size)
    output = AdaptiveLIFNodes(
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
    self.output_monitor = output_monitor
    self.add_monitor(output_monitor, name='output_monitor')
    self.add_layer(input, name='input')
    self.add_layer(output, name='output')

    ## Connections ##
    in_out_wfeat = Weight(name='in_out_weight_feature', value=w_in_out)
    in_out_conn = MulticompartmentConnection(
      source=input, target=output,
      device=device, pipeline=[in_out_wfeat],
    )
    out_out_wfeat = Weight(name='out_out_weight_feature', value=w_out_out)
    out_out_conn = MulticompartmentConnection(
      source=output, target=output,
      device=device, pipeline=[out_out_wfeat],
    )
    self.add_connection(in_out_conn, source='input', target='output')
    self.add_connection(out_out_conn, source='output', target='output')
    self.weights = in_out_wfeat
    self.w_mask = in_out_wfeat.value != 0

    ## Migrate ##
    self.to(device)

    ## STDP-RL Parameters ##
    self.eligibility = torch.zeros(in_size, out_size, device=device)
    self.lr = lr
    self.decay = decay
    self.wmin = wmin
    self.wmax = wmax


    ## Q-Learning parameters ##
    self.alpha = alpha
    self.gamma = gamma
    self.q_table = {}

  def STDP_RL(self, in_spikes, out_spikes, next_state, delta_Q):
    in_activity = in_spikes.squeeze().sum(dim=0)
    out_activity = out_spikes.squeeze().sum(dim=0)
    self.eligibility = torch.outer(in_activity, out_activity)

    # Update weights
    alpha = 0.002
    if np.abs(delta_Q) < 1e-4:
      r = 0
    else:
      r = np.sign(delta_Q)
    # self.weights.value[self.eligibility > 0] *= 0.99
    self.weights.value += alpha * r * self.eligibility
    self.weights.value[self.w_mask == 0] = 0
    self.weights.value *= (1-self.decay)
    self.weights.value += self.lr * r * self.eligibility
    torch.clamp(self.weights.value, self.wmin, self.wmax)
    print(self.weights.value.max())

  def update_table(self, state, action, next_state, reward):
    # Preprocess state for simpler indexing
    # Sum spikes -> get sorted indices
    action = action.item()
    if state not in self.q_table:
      self.q_table[state] = np.zeros(4)    # 4 moves
      self.q_table[state][action] = self.alpha * reward
      delta_Q = self.q_table[state][action]
    else:
      org_val = self.q_table[state][action]
      next_state_val = 0 if next_state not in self.q_table else max(self.q_table[next_state])
      self.q_table[state][action] *= (1 - self.alpha)  # Decay
      self.q_table[state][action] += self.alpha * (reward + self.gamma * next_state_val)  # Update
      delta_Q = self.q_table[state][action] - org_val
    return delta_Q

  def plot(self):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(self.weights.value, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax1)
    im = ax2.imshow(self.eligibility, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax2)
    ax1.set_title("Weights")
    ax2.set_title("Eligibility")
    plt.show()

  def plot_q_table(self, env):
    # Get max action for each position
    env_shape = (env.maze.width, env.maze.height)
    max_actions = np.zeros(env_shape)
    for state, actions in self.q_table.items():
      coords = state
      max_actions[coords[0], coords[1]] = np.argmax(actions)
    # Plot max actions with arrows
    fig, ax = plt.subplots()
    # ax.imshow(max_actions, cmap='hot', interpolation='nearest')
    for i in range(env_shape[0]):
      for j in range(env_shape[1]):
        if max_actions[i, j] == 0:
          ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
        elif max_actions[i, j] == 1:
          ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        elif max_actions[i, j] == 2:
          ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
        elif max_actions[i, j] == 3:
          ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    # Plot maze
    env.plot(ax)
    plt.show()

  # Plot move count for each tile in the maze
  def plot_move_count(self, history, env):
    fig, ax = plt.subplots()
    move_count = np.zeros((env.maze.width, env.maze.height, 4))
    for _, coords, action, _, _, _ in history:
      move_count[coords[0], coords[1], action] += 1
    # ax.imshow(max_actions, cmap='hot', interpolation='nearest')
    for i in range(env.maze.width):
      for j in range(env.maze.height):
        if move_count[i, j, 0] > 0:
          ax.text(j, i - 0.3, int(move_count[i, j, 0]), ha='center', va='center', color='blue')
        if move_count[i, j, 1] > 0:
          ax.text(j + 0.3, i, int(move_count[i, j, 1]), ha='center', va='center', color='blue')
        if move_count[i, j, 2] > 0:
          ax.text(j, i + 0.3, int(move_count[i, j, 2]), ha='center', va='center', color='blue')
        if move_count[i, j, 3] > 0:
          ax.text(j - 0.3, i, int(move_count[i, j, 3]), ha='center', va='center', color='blue')

    # Plot maze
    env.plot(ax)
    plt.show()

# Select action using epsilon-greedy policy
def select_action(assoc_spikes, sim_time, out_size, eps, model, env):
  motor_pop_size = out_size // env.num_actions

  # Select action from policy net
  if random.random() > eps:
    ## Pass association spikes through model ##
    model.run(inputs={'input': assoc_spikes}, time=sim_time)
    out_spikes = model.output_monitor.get('s')
    if torch.max(out_spikes) == 0:
      action = np.random.choice(env.num_actions)    # Random action if no spikes
      out_spikes = torch.zeros(sim_time, out_size)
      motor_pop_range = (action * motor_pop_size, action * motor_pop_size + motor_pop_size)
      motor_pop_spikes = torch.rand(sim_time, motor_pop_size) < 0.05
      out_spikes[:, motor_pop_range[0]:motor_pop_range[1]] = motor_pop_spikes
      print("no spikes!")
    else:
      action = torch.argmax(out_spikes.reshape(sim_time, env.num_actions, motor_pop_size).sum(0).sum(1))
    model.reset_state_variables()
    return torch.tensor([action]), out_spikes.squeeze(), True   # action, out spikes, chosen_by_policy

  # Select random action (exploration)
  else:
    # Generate artificial out_spikes
    action = np.random.choice(env.num_actions)
    out_spikes = torch.zeros(sim_time, out_size)
    motor_pop_range = (action*motor_pop_size, action*motor_pop_size+motor_pop_size)
    motor_pop_spikes = torch.rand(sim_time, motor_pop_size) < 0.1
    out_spikes[:, motor_pop_range[0]:motor_pop_range[1]] = motor_pop_spikes
    return torch.tensor([action]), out_spikes, False  # action, out spikes, chosen_by_policy


def run_episode(env, model, in_size, out_size, motor_pop_size, max_steps, sim_time, eps=0, learning=False, device='cpu'):
  # Initialize the environment and get its state
  state, coords, _ = env.reset()
  state = state[:, 0:in_size]
  history = []
  for t in count():
    action, out_spikes, chosen_by_policy = select_action(state, sim_time, out_size, eps, model, env)
    observation, reward, terminated, next_state_coords, _ = env.step(action)
    next_state = torch.tensor(observation, dtype=torch.bool, device=device)
    delta_Q = model.update_table(coords, action, next_state_coords, reward)

    # Update history
    history.append((state.numpy(), coords, action.item(), reward, out_spikes.numpy(), delta_Q))

    # Perform one step of the optimization
    if learning:
      model.STDP_RL(state, out_spikes, next_state_coords, delta_Q)

    # Break if terminated or max_steps reached
    if terminated or t >= max_steps:
      # env.animate_history(history, motor_pop_size=motor_pop_size)
      # model.plot()
      # plt.show()
      break

    # Move to the next state
    state = next_state
    coords = next_state_coords
    state = state[:, 0:in_size]

  return history


def record_episode(env, model, in_size, out_size, max_steps, sim_time, eps, learning, device, filename):
  history = run_episode(env, model, in_size, out_size, max_steps, sim_time, eps, learning, device)  # eps = 0 -> no exploration
  env.animate_history(history, filename)
  plt.clf()


def train_STDP_RL(recalled_memories_sorted, env_width, env_height, max_total_steps, max_steps_per_ep, eps_start,
                  eps_end, decay_intensity, in_size, out_size, motor_pop_size, sim_time, hyper_params,
                  env_trace_length, env_path, alpha, gamma, lr, decay, wmin, wmax, device='cpu',
                  plot=False, save=True):

  ## Init model & maze ##
  w_in_out = torch.rand((in_size, out_size))*2
  w_in_out = sparsify(w_in_out, 0.7)
  w_out_out = -torch.ones((out_size, out_size))
  # Inhibit other motor pop, excite own motor pop
  for i in range(4):
    w_out_out[i*motor_pop_size:(i+1)*motor_pop_size, i*motor_pop_size:(i+1)*motor_pop_size] = 0.1
  model = STDP_RL_Model(in_size, out_size, hyper_params, w_in_out, w_out_out, alpha, gamma,
                        lr, decay, wmin, wmax, device)
  env = Grid_Cell_Maze_Environment(width=env_width, height=env_height, trace_length=env_trace_length,
                                   in_spikes=recalled_memories_sorted,
                                   load_from=env_path)

  ## Training loop ##
  episode_durations = []
  move_count = np.zeros((env.maze.width, env.maze.height, 4))
  universal_history = []
  episodes = 0
  total_steps = 0
  print(env.maze)
  while total_steps < max_total_steps:
    eps = eps_end + (eps_start - eps_end) * math.exp(-decay_intensity * total_steps / (max_total_steps))
    history = run_episode(env, model, in_size, out_size, motor_pop_size, max_steps_per_ep, sim_time, eps=eps, learning=True, device=device)
    universal_history.append(history)
    # update move count
    for _, coords, action, _, _, _ in history:
      move_count[coords[0], coords[1], action] += 1
    total_steps += len(history)
    episode_durations.append(len(history))
    print(f"Episode {episodes} lasted {len(history)} steps, eps = {round(eps, 2)} total steps = {total_steps}, "
          f"avg reward = {round(sum([h[3] for h in history])/len(history), 2)} "
          f"avg weight = {round(model.weights.value.mean().item(), 5)}")
    episodes += 1

  ## Post-training recording ##
  if save:
    with open('Output/universal_history.pkl', 'wb') as f:
      pkl.dump(universal_history, f)
  # if plot:
    # for i, h in enumerate(universal_history):
    #   env.animate_history(h, file_name=f"Output/ep{i}.gif" ,motor_pop_size=motor_pop_size)
    # env.animate_history(history, motor_pop_size=motor_pop_size)
    # model.plot()
    # plt.show()
    # model.plot_q_table(env)
    # # Plot confusion
    # model.plot_move_count(move_count, env)

  ## Plot ##
  if plot:
    # Episodes
    plt.plot(episode_durations)
    plt.title("Episode durations")
    plt.ylabel("Duration")
    plt.xlabel("Episode")
    plt.show()
    model.plot()

    # Q-table
    model.plot_q_table(env)

    # Confusion
    model.plot_move_count([h for hh in universal_history for h in hh], env)

  ## Return score ##
  i = int(len(episode_durations) * 0.2)
  score = np.average(episode_durations[-2:])
  return score