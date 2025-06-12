from collections import defaultdict
from itertools import count

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from Grid_Cells import GC_Population
from STDP_Q_Learning import STDP_Q_Learning
from Reservoir import Reservoir, ANDReservoir
from Environment import Grid_Cell_Maze_Environment


# Generate grid cell activity for each coordinate in the environment
def grid_cell_activity_generator(maze_size, gc_pop: GC_Population):
  # Generate the spike activity for each coordinate in the environment
  x_range, y_range = maze_size
  activity = np.zeros((x_range, y_range, gc_pop.n_cells))
  for i in range(x_range):
    for j in range(y_range):
      pos = (i, j)
      a = gc_pop.activity(pos)
      activity[i, j] = a
  return activity


# Convert grid cell activity to spike trains
# Return spike trains of shape (x, y, n_cells, sim_time)
def spike_train_generator(gc_activity: np.array, sim_time, max_firing_rates):
  # Note: gc_activity values in range of values [0, 1]
  time_denominator = 1000  # working in ms
  x_range, y_range, n_cells = gc_activity.shape
  spike_trains = np.zeros((*gc_activity.shape, sim_time))
  for i in range(x_range):
    for j in range(y_range):
      for k in range(n_cells):
        activity = gc_activity[i, j, k]   # in range [0, 1]
        max_freq = max_firing_rates[k]    # max firing rate for this grid cell
        spike_rate = activity * max_freq / time_denominator  # spike rate per ms
        spike_train = np.zeros(sim_time)
        if spike_rate != 0:
          step_size = int(1 / spike_rate) # number of ms between spikes
          spike_train[::step_size] = 1
        spike_trains[i, j, k] = spike_train
  return spike_trains


# Calculate the diversity of the grid cell spike trains
# Diversity = avg. difference in # of spikes per grid cell between all pairs of coordinates
def diversity(spike_trains: np.array):
  # spike_trains is a 3D numpy array of shape (x, y, n_cells, sim_time)
  x_range, y_range, n_cells, sim_time = spike_trains.shape
  correlations = np.zeros((x_range*y_range, x_range*y_range))
  for i in range(x_range):
    for j in range(y_range):
      for n in range(x_range):
        for m in range(y_range):
          s1 = spike_trains[i, j].sum(axis=1)  # total number of spikes for each cell
          s2 = spike_trains[n, m].sum(axis=1)
          corr = np.sum(np.abs(s1 - s2))       # Pairwise difference between spike trains
          corr /= n_cells                      # Normalize by number of cells (avg. difference in spikes)
          idx = (i*x_range + j, n*x_range + m)
          correlations[idx] = corr
  return correlations


# Determine which pre-synaptic neuron is most relevant
# Return index of top n most relevant (non-zero) neurons
def relevant_neurons(spike_train: torch.tensor, n: int):
  total_spikes = spike_train.sum(axis=1)
  top_n_indices = np.argpartition(total_spikes, -n)[-n:]
  sorted_indices = top_n_indices[np.argsort(total_spikes[top_n_indices])][::-1]
  returned_indices = []
  firing_rates = []
  for ind in sorted_indices:
    if total_spikes[ind] > 4:
      returned_indices.append(ind)
      firing_rates.append(total_spikes[ind])
  return returned_indices, firing_rates


def generate_weights(in_size, out_size, sparsity, range):
  wmin, wmax = range
  # w = np.random.uniform(0, 1, (in_size, out_size))
  # sparsity_mask = np.random.choice([0, 1], w.shape, p=[1-sparsity, sparsity])
  # w = np.random.choice([0, 1], size=(in_size, out_size), p=[1-sparsity, sparsity])
  w = np.zeros(in_size * out_size)
  num_ones = int(sparsity * in_size * out_size)
  w[:num_ones] = 1
  np.random.shuffle(w)
  w *= wmax
  w = w.reshape(in_size, out_size)
  return w


# # One one synapse per out neuron
# def generate_res_out_weights(in_size, out_size):
#   w = np.zeros((in_size, out_size))
#   for i in range(in_size):
#     j = np.random.randint(0, out_size)
#     w[i, j] = 1
#   return w


def run(parameters: dict):
  ## Run Parameters ##
  PLOT = parameters['plot']
  ANIMATE_TRAINING = parameters['animate_training']
  MAZE_SIZE = parameters['maze_size']
  SCALES = parameters['scales']
  ROTATIONS = parameters['rotations']
  NUM_MODULES = parameters['num_modules']
  OFFSETS_PER_MODULE = parameters['offsets_per_module']
  GLOBAL_SCALE = parameters['global_scale']
  SHARPNESSES = parameters['sharpness']
  SIM_TIME = parameters['sim_time']
  AND1_SIZE = parameters['AND1_size']
  AND2_SIZE = parameters['AND2_size']
  HYPERPARAMS = parameters['hyperparams']
  SPARSITIES = parameters['sparsities']
  RANGES = parameters['ranges']
  ALPHA = parameters['alpha']
  GAMMA = parameters['gamma']
  DECAY = parameters['decay']
  LR = parameters['lr']
  TRACE_LENGTH = parameters['trace_length']
  ENV_PATH = parameters['env_path']
  MAX_STEPS = parameters['max_steps']
  NUM_EPISODES = parameters['episodes']

  ## Grid Cell activity generator ##
  gc_pop = GC_Population(NUM_MODULES, OFFSETS_PER_MODULE, GLOBAL_SCALE, SCALES, ROTATIONS, SHARPNESSES)
  gc_activity = grid_cell_activity_generator(MAZE_SIZE, gc_pop)

  ## Convert Grid Cell activity to spike trains ##
  gc_spike_trains = spike_train_generator(gc_activity, sim_time=1000, max_firing_rates=gc_pop.max_firing_rates)

  ## Push spike trains through association area ##
  n_cells = gc_pop.n_cells
  w_in_AND1 = generate_weights(n_cells, AND1_SIZE, SPARSITIES['in_AND1'], RANGES['in_AND1'])
  w_AND1_AND2 = generate_weights(AND1_SIZE, AND2_SIZE, SPARSITIES['AND1_AND2'], RANGES['AND1_AND2'])
  reservoir = ANDReservoir(
             in_size=n_cells,
             AND1_size=AND1_SIZE,
             AND2_size=AND2_SIZE,
             w_in_AND1=w_in_AND1,
             w_AND1_AND2=w_AND1_AND2,
             hyper_params=HYPERPARAMS,)
  AND1_spike_trains = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], 1000, AND1_SIZE)
  AND2_spike_trains = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], 1000, AND2_SIZE)
  for i in range(MAZE_SIZE[0]):
    for j in range(MAZE_SIZE[1]):
      AND1_spikes, AND2_spikes = reservoir.get_spikes(gc_spike_trains[i, j], sim_time=1000)  # Run for 1 second
      AND1_spike_trains[i, j] = AND1_spikes.squeeze(1)  # (time, AND1)
      AND2_spike_trains[i, j] = AND2_spikes.squeeze(1)  # (time, AND2)

  # Plot reservoir spike trains
  # Also calculate the diversity in reservoir activity
  if PLOT:
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(MAZE_SIZE[0]*2+1, MAZE_SIZE[1]*2+1)
    fp_ax = fig.add_subplot(gs[:MAZE_SIZE[0], :MAZE_SIZE[1]])
    gc_pop.plot_peaks([-1, MAZE_SIZE[0]], [-1, MAZE_SIZE[1]], fig=fig, ax=fp_ax, ) # contours=True, pos=(0, 1))

    # Plot grid cell activity (Bottom left)
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        ax = fig.add_subplot(gs[MAZE_SIZE[0]+i+1, j])
        ax.imshow(gc_spike_trains[i, j], aspect='auto', cmap='binary', interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot AND1 activity (Top right)
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        ax = fig.add_subplot(gs[i, MAZE_SIZE[1]+j+1])
        ax.imshow(AND1_spike_trains[i, j].T, aspect='auto', cmap='binary', interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot AND2 activity (Bottom right)
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        ax = fig.add_subplot(gs[MAZE_SIZE[0]+i+1, MAZE_SIZE[1]+j+1])
        ax.imshow(AND2_spike_trains[i, j].T, aspect='auto', cmap='binary', interpolation=None)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("spike_trains.png", dpi=1000)
    exit()
    # plt.show()

  ## Analyze GC activity ##
  rel_neurons = {}
  used_neurons = defaultdict(list)
  for i in range(MAZE_SIZE[0]):   # Calculate relevant neurons for each position
    for j in range(MAZE_SIZE[1]):
      top_neurons, firing_rates = relevant_neurons(gc_spike_trains[i, j], 50)
      rel_neurons[(i, j)] = (top_neurons, firing_rates)
      # print(f"Position: ({i, j}), \n\tTop Neurons: {top_neurons}, \n\tFiring Rates: {firing_rates}")
      if len(top_neurons) < 2:
        print(f"Position: ({i, j}) has only {len(top_neurons)} relevant neurons. \n\tTop Neurons: {top_neurons}, \n\tFiring Rates: {firing_rates}")
      # Find overlaps in relevant neurons
      for neuron in top_neurons:
        used_neurons[neuron].append((i, j))

  # Check if any two positions share 2 or more relevant neurons
  rel_list = list(rel_neurons.items())
  for i in range(len(rel_list)):
    for j in range(i+1, len(rel_list)):
      pos1, data1 = rel_list[i]
      pos2, data2 = rel_list[j]
      if pos1 != pos2:
        overlap = set(data1[0]) & set(data2[0])
        if len(overlap) >= 2:
          print(f"Positions {pos1} and {pos2} share {overlap} neurons.")

  ## Perform Q-Learning ##
  w_AND2_out = generate_weights(AND2_SIZE, 4, SPARSITIES['AND2_out'], RANGES['AND2_out'])
  w_out_out = generate_weights(4, 4, SPARSITIES['out_out'], RANGES['out_out'])
  model = STDP_Q_Learning(
    in_size=AND2_SIZE,
    out_size=4,
    w_exc_out=w_AND2_out,
    w_out_out=w_out_out,
    alpha=ALPHA,
    gamma=GAMMA,
    num_actions=4,
    wmin=RANGES['AND2_out'][0],
    wmax=RANGES['AND2_out'][1],
    decay=DECAY,
    lr=LR,
    hyper_params=HYPERPARAMS,
  )

  env = Grid_Cell_Maze_Environment(
    width=MAZE_SIZE[0],
    height=MAZE_SIZE[1],
    in_spikes=AND2_spike_trains,
    trace_length=TRACE_LENGTH,
    load_from=ENV_PATH
  )

  if ANIMATE_TRAINING:
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(3, 3)
    maze_ax = fig.add_subplot(gs[0:3, -2:])   # top-to-bottom right
    weights_ax = fig.add_subplot(gs[0, 0])    # top-left
    res_spikes_ax = fig.add_subplot(gs[1, 0])     # middle-left
    out_spikes_ax = fig.add_subplot(gs[2, 0])     # bottom-left
    maze_ax.set_title("Maze")
    weights_ax.set_title("Res-Out Weights")
    res_spikes_ax.set_title("Res Spikes")
    out_spikes_ax.set_title("Out Spikes")

  def run_episode(animate=False):
    # state: spike trains of shape (exc+inh, time)
    state, coords, _ = env.reset()
    out_spikes = np.zeros((4, SIM_TIME))
    history = []
    for t in count():
      if animate:
        maze_ax.clear()
        weights_ax.clear()
        res_spikes_ax.clear()
        out_spikes_ax.clear()
        model.plot_weights(ax=weights_ax)
        model.plot_spikes(ax=res_spikes_ax, spikes=state)
        model.plot_spikes(ax=out_spikes_ax, spikes=out_spikes)
        env.plot(coords, q_table=model.q_table, ax=maze_ax)
        plt.tight_layout()
        plt.pause(0.00001)
      action, out_spikes = model.select_action(state, SIM_TIME)
      new_state, reward, terminated, new_coords = env.step(action)

      # ## TODO: Spaghetti code!
      # if len(history) > 5:
      #   avg_reward = np.mean([h[2] for h in history[-5:]])
      #   modular_learning_rate = 0.1 * (((avg_reward - 1)**2) / 4)
      #   model.lr = modular_learning_rate
      # ## TODO: Spaghetti code!

      delta_Q = model.Q_Learning(coords, action, reward, new_coords)   # Alternatively with new_state rather than coords
      # delta_Q = model.Q_Learning(new_state, action, reward, new_state)
      model.STDP_RL(reward, state, out_spikes)
      model.reset_state_variables()
      history.append((state, action, reward, new_state, delta_Q))
      print(f"Step {t+1}/{MAX_STEPS} - Reward: {reward:.2f}")
      if terminated or t >= MAX_STEPS:
        break
      state = new_state
      coords = new_coords
    return history

  # Train model
  universal_history = []
  for episode in range(NUM_EPISODES):
    history = run_episode(ANIMATE_TRAINING)
    print(f"Episode {episode+1}/{NUM_EPISODES} - Steps: {len(history)}")
    universal_history.append(history)


if __name__ == '__main__':
  # primes = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,])
  # num_modules = 5    # First 5 primes
  # offsets_per_module = 5  # 5 GC per module
  # global_scale = 0.25  # How much to scale the entire grid cell system
  # NUM_CELLS = num_modules * offsets_per_module**2
  #
  # ## Scale entire system ##
  # primes = primes * global_scale
  #
  # ## Offsets ##
  # x_offsets = []
  # y_offsets = []
  # for i in range(num_modules):
  #   p = primes[i]
  #   offset_step_size = p / offsets_per_module
  #   base_x_offsets = []
  #   for j in range(1, offsets_per_module+1):
  #     base_x_offsets.append(offset_step_size*j)
  #   base_y_offsets = base_x_offsets.copy()
  #   mod_x_offsets, mod_y_offsets = np.meshgrid(base_x_offsets, base_y_offsets)
  #   mod_x_offsets = mod_x_offsets.flatten()   # Transform into 1D arrays
  #   mod_y_offsets = mod_y_offsets.flatten()
  #   x_offsets.extend(mod_x_offsets)
  #   y_offsets.extend(mod_y_offsets)
  #
  # ## Scales ##
  # scales = np.ones(NUM_CELLS)
  # for i in range(num_modules):
  #   p = primes[i]
  #   scales[i*num_modules**2:(i+1)*num_modules**2] = p
  #
  # rotations = [0] * NUM_CELLS
  # sharpness = [1] * NUM_CELLS
  primes = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,])
  np.random.seed(1)
  p = {
    'plot': True,
    'animate_training': False,
    'maze_size': (5, 5),
    'num_modules': 5,
    'offsets_per_module': 3,
    'scales': primes,
    'global_scale': 0.5,
    'rotations': [0, 1.5, 2],
    'sharpness': 1.5,    # Should *not* go below 1
    'sim_time': 1000, # ms
    'AND1_size': 100,
    'AND2_size': 100,
    'hyperparams': {
      "AND1_refrac": 1,
      "AND1_reset": -64,   # Base
      "AND1_tc_decay": 12,  # AND decay for 50ms interval @ 11mv threshold
      "AND1_tc_theta_decay": 10_000,
      "AND1_theta_plus": 0,
      "AND1_thresh": -45,  # 21mv threshold (~3-way AND)
      "AND2_refrac": 1,
      "AND2_reset": -64,
      "AND2_tc_decay": 12,  # AND decay for 100ms interval @ 11mv threshold
      "AND2_tc_theta_decay": 10_000,
      "AND2_theta_plus": 0,
      "AND2_thresh": -53,  # 11mv threshold (2-way AND)
      "refrac_out": 1,
      "reset_out": -64,
      "tc_decay_out": 35, # AND decay for 100ms interval @ 11mv threshold
      "tc_theta_decay_out": 10_000,
      "theta_plus_out": 0,
      "thresh_out": -53,  # 11mv threshold
    },
    'ranges': {
      'in_AND1': (0, 10),
      'AND1_AND2': (0, 10),
      'AND2_out': (0, 10),
      'out_out': (-10, -10)
    },
    'sparsities': {
      'in_AND1': 0.1,
      'AND1_AND2': 0.05,
      'AND2_out': 0.05,
      'out_out': 1
    },
    'alpha': 0.1,   # Q-Table learning rate
    'gamma': 0.9,   # Q-Table discount factor (how much future rewards are discounted)
    'decay': 0.1,   # Synaptic decay (UNUSED)
    'lr': 0.1,      # Weight update learning rate
    'trace_length': 0,
    'env_path': 'env.pkl',
    'max_steps': 1000,
    'episodes': 100,
  }
  run(p)