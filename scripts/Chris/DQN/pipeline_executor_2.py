from collections import defaultdict
from itertools import count
import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from Grid_Cells import GC_Population
from STDP_Q_Learning import STDP_Q_Learning
from Reservoir import Reservoir
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
def relevant_neurons(spike_train: torch.tensor, threshold: int = 4):
  total_spikes = spike_train.sum(axis=1)
  top_indices = np.where(total_spikes > threshold)[0]
  sorted_indices = top_indices[np.argsort(total_spikes[top_indices])][::-1]
  returned_indices = []
  firing_rates = []
  for ind in sorted_indices:
    if total_spikes[ind] > threshold:
      returned_indices.append(ind)
      firing_rates.append(total_spikes[ind])
  return returned_indices, firing_rates


# Generate random weights for the grid cell to reservoir connections
def generate_grid_out_weights(in_size, out_size, sparsity, w_range):
  wmin, wmax = w_range
  w = np.zeros(in_size * out_size)
  num_ones = int(sparsity * in_size * out_size)
  w[:num_ones] = 1
  np.random.shuffle(w)
  w *= wmax   # TODO: NOTE: currently ignores min range, all synapses start with same strength
  w = w.reshape(in_size, out_size)
  return w

# Generate random weights for the reservoir to output connections
# Cumulative sum of weights for each pre-synaptic neuron (row) is equal to pre_synaptic_magnitude
def generate_res_out_weights(in_size, out_size, pre_synaptic_magnitude, noise_scale=1):
  # Base array of evenly distributed weights
  base_val = pre_synaptic_magnitude / out_size
  w = np.full((in_size, out_size), base_val)
  # Add noise to weights
  noise = 1 + noise_scale * (2 * np.random.rand(in_size, out_size) - 1)
  w *= noise
  # Renormalize
  sum = w.sum(axis=1, keepdims=True)
  w *= pre_synaptic_magnitude / sum
  return w


def run(parameters: dict):
  ## Run Parameters ##
  PLOT = parameters['plot']
  ANIMATE_TRAINING = parameters['animate_training']
  SAVE_FILE = parameters['save_file']
  LOAD_FROM_FILE = parameters['load_from_file']
  MAZE_SIZE = parameters['maze_size']
  # NUM_CELLS = parameters['num_cells']
  # X_OFFSETS = parameters['x_offsets']
  # Y_OFFSETS = parameters['y_offsets']
  SCALES = parameters['scales']
  ROTATIONS = parameters['rotations']
  NUM_MODULES = parameters['num_modules']
  OFFSETS_PER_MODULE = parameters['offsets_per_module']
  GLOBAL_SCALE = parameters['global_scale']
  SHARPNESSES = parameters['sharpness']
  SIM_TIME = parameters['sim_time']
  EXC_SIZE = parameters['exc_size']
  INH_SIZE = parameters['inh_size']
  OUT_SIZE = parameters['out_size']
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

  if not LOAD_FROM_FILE:
    ## Grid Cell activity generator ##
    # gc_m = GC_Module(NUM_CELLS, X_OFFSETS, Y_OFFSETS, ROTATIONS, SCALES, SHARPNESSES)
    gc_pop = GC_Population(NUM_MODULES, OFFSETS_PER_MODULE, GLOBAL_SCALE, SCALES, ROTATIONS, SHARPNESSES)
    gc_activity = grid_cell_activity_generator(MAZE_SIZE, gc_pop)

    ## Convert Grid Cell activity to spike trains ##
    gc_spike_trains = spike_train_generator(gc_activity, sim_time=1000, max_firing_rates=gc_pop.max_firing_rates)

    ## Push spike trains through association area ##
    n_cells = gc_pop.n_cells
    w_in_exc = generate_grid_out_weights(n_cells, EXC_SIZE, SPARSITIES['in_exc'], RANGES['in_exc'])
    w_in_inh = generate_grid_out_weights(n_cells, INH_SIZE, SPARSITIES['in_inh'], RANGES['in_inh'])
    w_exc_exc = generate_grid_out_weights(EXC_SIZE, EXC_SIZE, SPARSITIES['exc_exc'], RANGES['exc_exc'])
    w_exc_inh = generate_grid_out_weights(EXC_SIZE, INH_SIZE, SPARSITIES['exc_inh'], RANGES['exc_inh'])
    w_inh_exc = -generate_grid_out_weights(INH_SIZE, EXC_SIZE, SPARSITIES['inh_exc'], RANGES['inh_exc'])
    w_inh_inh = -generate_grid_out_weights(INH_SIZE, INH_SIZE, SPARSITIES['inh_inh'], RANGES['inh_inh'])
    reservoir = Reservoir(
               in_size=n_cells,
               exc_size=EXC_SIZE,
               inh_size=INH_SIZE,
               w_in_exc=w_in_exc,
               w_in_inh=w_in_inh,
               w_exc_exc=w_exc_exc,
               w_exc_inh=w_exc_inh,
               w_inh_exc=w_inh_exc,
               w_inh_inh=w_inh_inh,
               hyper_params=HYPERPARAMS,)
    res_spike_trains = torch.zeros(MAZE_SIZE[0], MAZE_SIZE[1], 1000, EXC_SIZE+INH_SIZE)
    for i in range(MAZE_SIZE[0]):
      for j in range(MAZE_SIZE[1]):
        exc_spikes, inh_spikes = reservoir.get_spikes(gc_spike_trains[i, j], sim_time=1000)  # Run for 1 second
        res_spike_trains[i, j] = torch.concat((exc_spikes, inh_spikes), dim=2).squeeze(1)  # (time, exc+inh)

    # Plot reservoir spike trains
    # Also calculate the diversity in reservoir activity
    if PLOT:
      fig = plt.figure(figsize=(11, 11))
      gs = fig.add_gridspec(MAZE_SIZE[0]*2+1, MAZE_SIZE[1]*2+1)
      fp_ax = fig.add_subplot(gs[:MAZE_SIZE[0], :MAZE_SIZE[1]])
      gc_pop.plot_peaks([-1, MAZE_SIZE[0]], [-1, MAZE_SIZE[1]], fig=fig, ax=fp_ax, ) # contours=True, pos=(0, 1))
      fp_ax.grid(True)

      for i in range(MAZE_SIZE[0]):
        for j in range(MAZE_SIZE[1]):
          ax = fig.add_subplot(gs[MAZE_SIZE[0]+i+1, j])
          ax.imshow(gc_spike_trains[i, j], aspect='auto', cmap='binary', interpolation=None)
          ax.set_xticks([])
          ax.set_yticks([])
      for i in range(MAZE_SIZE[0]):
        for j in range(MAZE_SIZE[1]):
          ax = fig.add_subplot(gs[MAZE_SIZE[0]+i+1, j+MAZE_SIZE[1]+1])
          ax.imshow(res_spike_trains[i, j].T, aspect='auto', cmap='binary', interpolation=None)
          ax.set_xticks([])
          ax.set_yticks([])

      plt.savefig("spike_trains.png", dpi=1000)
      plt.show()

    ## Analyze GC activity ##
    print("################")
    print("## Grid Cells ##")
    print("################\n")
    rel_neurons = {}
    used_neurons = defaultdict(list)
    for i in range(MAZE_SIZE[0]):   # Calculate relevant neurons for each position
      for j in range(MAZE_SIZE[1]):
        top_neurons, firing_rates = relevant_neurons(gc_spike_trains[i, j], threshold=4)
        rel_neurons[(i, j)] = (top_neurons, firing_rates)
        print(f"Position: ({i, j}), \n\tTop GCs: {top_neurons}, \n\tFiring Rates: {firing_rates}")
        # Find overlaps in relevant neurons
        for neuron in top_neurons:
          used_neurons[neuron].append((i, j))
    # for neuron, positions in used_neurons.items():
    #   print(f"Neuron {neuron} is relevant in positions {positions}")

    # Check if any two positions share 2 or more relevant neurons
    rel_list = list(rel_neurons.items())
    for i in range(len(rel_list)):
      for j in range(i+1, len(rel_list)):
        pos1, data1 = rel_list[i]
        pos2, data2 = rel_list[j]
        if pos1 != pos2:
          overlap = set(data1[0]) & set(data2[0])
          if len(overlap) >= 2:
            print(f"Positions {pos1} and {pos2} share {overlap} Grid Cells.")

    ## Analyze Reservoir activity ##
    print("\n#####################")
    print("## Reservoir Cells ##")
    print("#####################\n")
    rel_neurons = {}
    used_neurons = defaultdict(list)
    for i in range(MAZE_SIZE[0]):  # Calculate relevant neurons for each position
      for j in range(MAZE_SIZE[1]):
        # NOTE: res_spike_trains is of shape (time, exc+inh), so we need to transpose it
        top_neurons, firing_rates = relevant_neurons(res_spike_trains[i, j].T.numpy(), threshold=4)
        rel_neurons[(i, j)] = (top_neurons, firing_rates)
        print(f"Position: ({i, j}), \n\tTop Res Neurons: {top_neurons}, \n\tFiring Rates: {firing_rates}")
        # Find overlaps in relevant neurons
        for neuron in top_neurons:
          used_neurons[neuron].append((i, j))
    # for neuron, positions in used_neurons.items():
    #   print(f"Neuron {neuron} is relevant in positions {positions}")

    # Check if any two positions share 2 or more relevant neurons
    rel_list = list(rel_neurons.items())
    for i in range(len(rel_list)):
      for j in range(i + 1, len(rel_list)):
        pos1, data1 = rel_list[i]
        pos2, data2 = rel_list[j]
        if pos1 != pos2:
          overlap = set(data1[0]) & set(data2[0])
          if len(overlap) >= 2:
            print(f"Positions {pos1} and {pos2} share {overlap} neurons.")

    if SAVE_FILE:
      with open(f"saves/{SAVE_FILE}", "wb") as f:
        pkl.dump(res_spike_trains, f)

  else:
    with open(f"saves/{LOAD_FROM_FILE}", "rb") as f:
      res_spike_trains = pkl.load(f)

  ## Perform Q-Learning ##
  exit()
  w_exc_out = generate_res_out_weights(EXC_SIZE + INH_SIZE, OUT_SIZE, 10)    # TODO: Temporary manual weight generation
  w_out_out = np.zeros((OUT_SIZE, OUT_SIZE))
  model = STDP_Q_Learning(
    in_size=EXC_SIZE+INH_SIZE,
    out_size=OUT_SIZE,
    w_exc_out=w_exc_out,
    w_out_out=w_out_out,
    alpha=ALPHA,
    gamma=GAMMA,
    num_actions=4,
    wmin=RANGES['exc_out'][0],
    wmax=RANGES['exc_out'][1],
    decay=DECAY,
    lr=LR,
    hyper_params=HYPERPARAMS,
  )

  env = Grid_Cell_Maze_Environment(
    width=MAZE_SIZE[0],
    height=MAZE_SIZE[1],
    in_spikes=res_spike_trains,
    trace_length=TRACE_LENGTH,
    load_from=ENV_PATH
  )

  if ANIMATE_TRAINING:
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 3)
    maze_ax = fig.add_subplot(gs[0:2, -2:])
    weights_ax = fig.add_subplot(gs[0, 0])
    spikes_ax = fig.add_subplot(gs[1, 0])

  def run_episode(animate=False):
    # state: spike trains of shape (exc+inh, time)
    state, coords, _ = env.reset()
    history = []
    for t in count():
      if animate:
        maze_ax.clear()
        weights_ax.clear()
        spikes_ax.clear()
        model.plot_weights(ax=weights_ax)
        model.plot_spikes(ax=spikes_ax, spikes=state)
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
  primes = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,])
  np.random.seed(2)
  p = {
    'plot': True,
    'animate_training': True,
    'save_file': "1000_res.pkl",
    'load_from_file': None,
    'maze_size': (5, 5),
    'num_modules': 5,
    'offsets_per_module': 3,
    'scales': primes,
    'global_scale': 0.25,
    'rotations': [0, np.pi/3, (np.pi/3)*2, np.pi],
    'sharpness': 1.25,    # Should *not* go below 1
    'sim_time': 1000, # ms
    'exc_size': 3000,
    'inh_size': 1,
    'out_size': 100*4,
    'hyperparams': {
      "exc_refrac": 1,
      "exc_reset": -64,   # Base
      "exc_tc_decay": 20,  # AND decay for 20ms interval @ 15mv threshold
      "exc_tc_theta_decay": 10_000,
      "exc_theta_plus": 0,
      "exc_thresh": -49,  #
      "inh_refrac": 1,
      "inh_reset": -64,
      "inh_tc_decay": 10_000,
      "inh_tc_theta_decay": 10_000,
      "inh_theta_plus": 0,
      "inh_thresh": -60,
      "refrac_out": 1,
      "reset_out": -64,  # Base
      "tc_decay_out": 12,   # AND decay for 20ms interval @ 11mv threshold
      "tc_theta_decay_out": 10_000,
      "theta_plus_out": 0,
      "thresh_out": -53,  # 11mv threshold
    },
    'ranges': {
      'in_exc': (0, 6.5),
      'in_inh': (0, 1),
      'exc_exc': (0, 1),
      'exc_inh': (0, 1),
      'inh_exc': (-1, 0),
      'inh_inh': (-1, 0),
      'exc_out': (0, 5),   # NOTE: Currently ignored
      'out_out': (-1, -1),

    },
    'sparsities': {
      'in_exc': 0.12,
      'in_inh': 0.0,
      'exc_exc': 0.0,
      'exc_inh': 0.0,
      'inh_exc': 0.0,
      'inh_inh': 0.0,
      'exc_out': 0.1,
      'out_out': 0.0,
    },
    'alpha': 0.1,   # Q-Table learning rate
    'gamma': 0.9,   # Q-Table discount factor (how much future rewards are discounted)
    'decay': 0.5,   # Synaptic decay (UNUSED)
    'lr': 0.1,      # Weight update learning rate
    'trace_length': 11,
    'env_path': 'env.pkl',
    'max_steps': 1000,
    'episodes': 100,
  }
  run(p)