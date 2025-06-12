import random
from collections import deque

import numpy as np
from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver
from matplotlib.pyplot import plot as plt

import pickle as pkl
import matplotlib.pyplot as plt
from torch import optim

class Maze_Environment():
  def __init__(self, width, height, trace_length=5):

    # Generate basic maze & solve
    self.width = width
    self.height = height
    self.maze = Maze(width=width, height=height, generator=DepthFirstSearchGenerator())
    self.solver = MazeSolver()
    self.path = self.solver.solve(self.maze)
    self.maze.path = self.path    # No idea why this is necessary
    self.agent_cell = self.maze.start_cell
    self.num_actions = 4
    self.path_history = []  # (state, reward, done, info)
    self.trace_length = trace_length

    # Add reward traces to cells
    self.reward_trace = self.calculate_reward_trace()

  # Paths of length 'trace_length' emanating from goal cell providing agent with reward
  def calculate_reward_trace(self):
    reward_trace = np.full((self.height, self.width), np.inf)
    goal = self.maze.end_cell.coordinates
    queue = deque([(goal, 0)])  # (cell coordinates, distance)
    visited = set()

    while queue:
      (x, y), dist = queue.popleft()
      if (x, y) in visited or dist > self.trace_length:
        continue
      visited.add((x, y))
      reward_trace[y, x] = dist

      for direction in Direction:
        if direction in self.maze[x, y].open_walls:
          neighbor = self.maze.neighbor(self.maze[x, y], direction)
          queue.append((neighbor.coordinates, dist + 1))

    # Normalize reward trace to go from large to small values
    max_dist = np.max(reward_trace[reward_trace != np.inf])
    reward_trace = max_dist - reward_trace
    reward_trace[reward_trace > self.trace_length] = 0  # Cut off at trace_length
    reward_trace[reward_trace == -np.inf] = 0  # Replace np.inf with 0
    return reward_trace.T

  def plot(self, agent_coords, q_table: dict, state_behavior: np.ndarray, ax=None):
    if ax is None:
      fig, ax = plt.subplots()

    # Transpose agent coordinates (just how the maze is stored)
    agent_coords = agent_coords[::-1]

    # Box around maze
    ax.plot([-0.5, self.width-1+0.5], [-0.5, -0.5], color='black')
    ax.plot([-0.5, self.width-1+0.5], [self.height-1+0.5, self.height-1+0.5], color='black')
    ax.plot([-0.5, -0.5], [-0.5, self.height-1+0.5], color='black')
    ax.plot([self.width-1+0.5, self.width-1+0.5], [-0.5, self.height-1+0.5], color='black')

    # Plot maze
    for row in range(self.height):
      for column in range(self.width):
        # Path
        cell = self.maze[column, row]  # Transpose maze coordinates (just how the maze is stored)
        if cell == self.maze.start_cell:
          ax.plot(row, column, 'go')
        elif cell == self.maze.end_cell:
          ax.plot(row, column,'bo')
        elif cell in self.maze.path:
          ax.plot(row, column, 'ro')

        # Walls
        if Direction.S not in cell.open_walls:
          ax.plot([row-0.5, row+0.5], [column+0.5, column+0.5], color='black')
        if Direction.E not in cell.open_walls:
          ax.plot([row+0.5, row+0.5], [column-0.5, column+0.5], color='black')

        # Table
        if q_table:
          if (column, row) in q_table:
            q_values = q_table[(column, row)]
          else:
            q_values = np.zeros(self.num_actions) # Actions are N, E, S, W
          ax.text(row, column-0.4, f'{q_values[0]:.2f}S', ha='center', va='center')
          ax.text(row+0.4, column, f'{q_values[1]:.2f}E', ha='center', va='center', rotation=90)
          ax.text(row, column+0.4, f'{q_values[2]:.2f}N', ha='center', va='center')
          ax.text(row-0.4, column, f'{q_values[3]:.2f}W', ha='center', va='center', rotation=90)

        if state_behavior is not None:
          activity = state_behavior[column, row]
          ax.text(row, column+0.2, f'{activity[0]}', ha='center', va='center')
          ax.text(row+0.2, column, f'{activity[1]}', ha='center', va='center')
          ax.text(row, column-0.2, f'{activity[2]}', ha='center', va='center')
          ax.text(row-0.2, column, f'{activity[3]}', ha='center', va='center')

    # Plot agent
    ax.plot(agent_coords[0], agent_coords[1], 'yo')

    return ax

  def reset(self):
    self.agent_cell = self.maze.start_cell
    self.path_history = []
    return self.agent_cell, {}

  # Takes action
  # Returns next state, reward, done, info
  def step(self, action):
    # Transform action into Direction
    if action == 0:
      action = Direction.N
    elif action == 1:
      action = Direction.E
    elif action == 2:
      action = Direction.S
    elif action == 3:
      action = Direction.W

    # Check if action runs into wall
    if action not in self.agent_cell.open_walls:
      self.path_history.append((self.agent_cell.coordinates, -1, False, action))
      return self.agent_cell, -1, False, {}

    # Move agent
    else:
      prev_cell = self.agent_cell
      self.agent_cell = self.maze.neighbor(self.agent_cell, action)
      if self.agent_cell == self.maze.end_cell:    # Check if agent has reached the end
        self.path_history.append((self.agent_cell.coordinates, 1, True, action))
        return self.agent_cell, 1, True, {}
      else:
        prev_trace = self.reward_trace[prev_cell.coordinates]
        new_trace = self.reward_trace[self.agent_cell.coordinates]
        reward = 1 if new_trace > prev_trace else 0
        self.path_history.append((self.agent_cell.coordinates, reward, False, action))
        return self.agent_cell, reward, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)


class Grid_Cell_Maze_Environment(Maze_Environment):
  def __init__(self, width, height, in_spikes, trace_length=5, load_from=None):
    if load_from is not None:
      with open(load_from, 'rb') as f:
        super().__init__(width, height, trace_length)
        obj_data = pkl.load(f)
        self.__dict__.update(obj_data.__dict__)
    else:
      super().__init__(width, height, trace_length)

    self.reward_trace = self.calculate_reward_trace()
    self.samples = in_spikes

  # Returns:
  # - Spike train of grid cell corresponding to agent's position
  # - Reset coordinates (x, y)
  # - info: (empty)
  def reset(self):
    cell, info = super().reset()
    return self.state_to_grid_cell_spikes(cell), cell.coordinates, info

  # Move in maze
  def step(self, action):
    obs, reward, done, info = super().step(action)
    coords = obs.coordinates
    obs = self.state_to_grid_cell_spikes(obs)
    return obs, reward, done, coords

  # Return stored spike trains at coordinate location
  def state_to_grid_cell_spikes(self, cell):
    return self.samples[cell.coordinates]


if __name__ == '__main__':
  np.random.seed(1)
  with open('saves/2000_res_7_7_maze.pkl', 'rb') as f:
    in_spikes = pkl.load(f)
  maze = Grid_Cell_Maze_Environment(7, 7, in_spikes, trace_length=0)
  maze.plot(agent_coords=(0, 0), q_table=None, state_behavior=None)
  plt.savefig('maze_7_7_plot.png')
  with open('maze_7_7.pkl', 'wb') as f:
    pkl.dump(maze, f)