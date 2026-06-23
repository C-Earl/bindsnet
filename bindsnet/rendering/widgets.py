from vispy import scene
import numpy as np
from abc import abstractmethod
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from .visuals import RasterTexture
from vispy.visuals.axis import Ticker


class FixedStepTicker(Ticker):
    """Major ticks at constant multiples of `step` so a sliding domain
    produces smoothly translating labels instead of relocated 'nice' ones."""
    def __init__(self, axis, step, anchors=None):
        super().__init__(axis, anchors=anchors)
        self.step = float(step)

    def _get_tick_frac_labels(self):
        domain = self.axis.domain
        flip = domain[1] < domain[0]
        lo, hi = (domain[1], domain[0]) if flip else (domain[0], domain[1])
        offset, scale, step = lo, (hi - lo), self.step

        first = np.ceil(lo / step) * step
        major = np.arange(first, hi + 1e-9, step)
        labels = ['%g' % x for x in major]

        minor_num = 4
        minstep = step / (minor_num + 1)
        minor = []
        for m in major:
            minor.extend(np.arange(m + minstep, m + step - 1e-9, minstep))
        minor = np.array(minor) if minor else np.array([])

        major_frac = (major - offset) / scale if scale else major - offset
        minor_frac = (minor - offset) / scale if (scale and minor.size) else minor
        use = (major_frac > -1e-4) & (major_frac < 1.0001)
        major_frac, labels = major_frac[use], [l for li, l in enumerate(labels) if use[li]]
        if minor.size:
            minor_frac = minor_frac[(minor_frac > -1e-4) & (minor_frac < 1.0001)]
        if flip:
            major_frac = 1 - major_frac
            minor_frac = 1 - minor_frac if minor.size else minor_frac
        return major_frac, minor_frac, labels


class AbstractWidget:
  def __init__(self):
    # self.view = scene.widgets.ViewBox()   # VisPy ViewBox for widget rendering
    self.grid = scene.widgets.Grid()    # Grid to hold widget view and axes (if applicable)
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
  def __init__(self, link_x=True, link_y=True):
    super().__init__()
    self.y_axis = scene.AxisWidget(orientation='left', axis_label_margin=62)
    self.grid.add_widget(self.y_axis, row=0, col=0).width_max = 95

    self.view = self.grid.add_view(row=0, col=1, border_color='white')
    self.view.camera = 'panzoom'

    self.x_axis = scene.AxisWidget(orientation='bottom')
    self.grid.add_widget(self.x_axis, row=1, col=1).height_max = 55

    if link_x: self.x_axis.link_view(self.view)   # follows camera
    if link_y: self.y_axis.link_view(self.view)


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


class RasterPlot(GraphPlotWidget):
  def __init__(self,
               layer_name: str,
               max_timesteps: int = 100):
    super().__init__()
    self.layer_name = layer_name
    self.layer = None           # Initialized after added to Application object
    self.max_timesteps = max_timesteps
    self.current_write_head = 0
    self.raster = None

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    self.raster = RasterTexture(
        layer_size=self.layer_size, width=self.max_timesteps,
        spike_tensor=self.layer.s,
    )
    self.view.add(self.raster)
    self.view.camera.rect = (0, 0, self.max_timesteps, self.layer_size)
    self.y_axis.axis.axis_label = "Neuron"
    self.x_axis.axis.axis_label = "Timestep"
    step = max(1, round(self.max_timesteps / 5 / 100) * 100) or 100  # e.g. 500 -> 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step,
      anchors=self.x_axis.axis.ticker._anchors,  # preserve label anchoring
    )

  def render(self, t):
    self.raster.migrate_spikes(t)
    # scene-x 0..max_timesteps is now linear in time -> relabel it as absolute time
    self.x_axis.axis.domain = (t - self.max_timesteps + 1, t)


  def get_history(self):
    return np.array(self.history, dtype=np.float32)
