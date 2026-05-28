from vispy import scene
from abc import abstractmethod
import numpy as np

from abc import abstractmethod


class AbstractWidget:
  def __init__(self,):
    # self.view = scene.widgets.ViewBox()   # VisPy ViewBox for widget rendering
    self.history = []       # List to store historical data for rendering. One element per time-step

  @abstractmethod
  def prime(self, network):
    pass

  @abstractmethod
  def render(self):
    pass

  @abstractmethod
  def get_history(self):
    pass

  def reset(self):
    self.history = []


# A plotting widget with x and y axis
class GraphPlotWidget(AbstractWidget):
  def __init__(self,):
    super().__init__()

    ### Create plot with x and y axes ###
    self.grid = scene.widgets.Grid()
    self.view = self.grid.add_view(
      row=0,
      col=1,
      border_color='white'
    )
    self.view.camera = 'panzoom'
    self.x_axis = scene.AxisWidget(
      orientation='bottom'
    )
    self.y_axis = scene.AxisWidget(
      orientation='left'
    )
    self.grid.add_widget(
      self.y_axis,
      row=0,
      col=0
    )
    self.grid.add_widget(
      self.x_axis,
      row=1,
      col=1
    )
    self.x_axis.link_view(self.view)
    self.y_axis.link_view(self.view)


class RasterPlot(GraphPlotWidget):
  def __init__(self,
               layer_name: str,
               max_timesteps: int = 100):
    super().__init__()
    self.layer_name = layer_name
    self.layer = None           # Initialized after added to Application object
    self.max_timesteps = max_timesteps

    # self.view.camera = 'panzoom'
    self.markers = scene.visuals.Markers(
      parent=self.view.scene
    )

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n

  def render(self, t):
    ### Extract spike data from layer ###
    spike_data = self.layer.s.cpu().numpy()
    spike_ids = np.where(spike_data > 0)[1]
    for sid in spike_ids:
        self.history.append([t, sid])

    if len(self.history) == 0:
        return

    ### Render ###
    points = np.array(self.history, dtype=np.float32)
    self.markers.set_data(
        points,
        face_color='white',
        size=4
    )
    self.view.camera.set_range(
        x=(max(0, t - self.max_timesteps), max(self.max_timesteps, t)),
        y=(0, self.layer_size)
    )

  def get_history(self):
    return np.array(self.history, dtype=np.float32)


class VoltagePlot(AbstractWidget):
  def __init__(self,
            width: float,
            height: float,
            x: float,
            y: float,
            layer_name: str,
            neuron_ids: list[int],
            max_timesteps: int = 100,
            y_range: tuple[float, float] = (-80.0, 40.0)
  ):

    super().__init__(width, height, x, y)
    self.layer_name = layer_name
    self.layer = None
    self.max_timesteps = max_timesteps
    self.neuron_ids = neuron_ids
    self.history = {}   # Dictionary mapping neuron ID to list of [timestep, voltage] pairs
    self.view.camera = 'panzoom'
    self.lines = {}
    self.y_range = y_range  # Plotted y-axis range

    # Initial camera range
    self.view.camera.set_range(
      x=(0, self.max_timesteps),
      y=(self.y_range[0], self.y_range[1])  # Typical membrane voltage range
    )

  def prime(self, network):

    self.layer = network.layers[self.layer_name]
    for nid in self.neuron_ids:
      self.history[nid] = []
      self.lines[nid] = scene.visuals.Line(
        parent=self.view.scene,
        width=2
      )

  def render(self, t):
    ### Extract voltage data from layer ###
    voltages = self.layer.v
    voltages = voltages.cpu().numpy().flatten()   # TODO: Make this more efficient/GPU-friendly?
    for nid in self.neuron_ids:
      v = voltages[nid]
      self.history[nid].append([t, v])

    all_values = []
    for nid in self.neuron_ids:
      points = np.array(
        self.history[nid],
        dtype=np.float32
      )

      if len(points) < 2:
        continue

      self.lines[nid].set_data(points)
      all_values.extend(points[:, 1])

    ### Render ###
    xmin = max(0, t - self.max_timesteps)
    xmax = max(self.max_timesteps, t)

    # Autoscale voltage range
    if len(all_values) > 0:
      ymin = min(all_values)
      ymax = max(all_values)
      padding = max(1.0, (ymax - ymin) * 0.1)
      self.view.camera.set_range(
        x=(xmin, xmax),
        y=(ymin - padding, ymax + padding)
      )

  def get_history(self, neuron_id):
    return np.array(
      self.history[neuron_id],
      dtype=np.float32
    )