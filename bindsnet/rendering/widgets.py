from vispy import scene
import numpy as np
import torch
import colorsys
from abc import abstractmethod
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from .visuals import RasterTexture, ScrollLine, FeatureMatrix
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


class VoltagePlot(GraphPlotWidget):
  def __init__(self,
               layer_name: str,
               neuron_ids: list[int],
               max_timesteps: int = 100,
               y_range: tuple[float, float] = (-80.0, 40.0)):
    super().__init__(link_x=False)               # x = time, relabeled manually like RasterPlot
    self.layer_name = layer_name
    self.layer = None
    self.neuron_ids = list(neuron_ids)
    self.max_timesteps = max_timesteps
    self.y_range = y_range
    self.lines = None

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    K = len(self.neuron_ids)
    idx = torch.as_tensor(self.neuron_ids, device=self.layer.v.device, dtype=torch.long)
    colors = np.array(
      [[*colorsys.hsv_to_rgb(i / max(K, 1), 0.9, 1.0), 1.0] for i in range(K)],
      dtype=np.float32)

    self.lines = ScrollLine(n_neurons=K, width=self.max_timesteps,
                            volt_getter=lambda: self.layer.v, idx=idx, colors=colors)
    self.view.add(self.lines)

    y0, y1 = self.y_range
    self.view.camera.rect = (0, y0, self.max_timesteps, y1 - y0)
    self.y_axis.axis.axis_label = "Voltage (mV)"
    self.x_axis.axis.axis_label = "Timestep"
    step = max(1, round(self.max_timesteps / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)

  def render(self, t):
    self.lines.migrate_voltages(t)
    self.x_axis.axis.domain = (t - self.max_timesteps + 1, t)

  def get_history(self):
    return None    # history lives on the GPU now; no CPU copy kept


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


class FeaturePlot(GraphPlotWidget):
  # language=rst
  """
  Abstract base for plotting a connection feature's ``value`` matrix as a live,
  GPU-resident heatmap (x = target neuron, y = source neuron, color = value),
  with a colorbar legend.

  Shared here: locating the :class:`AbstractFeature` in the network, driving the
  per-frame texture migration, and building the colorbar. Subclasses describe
  *how* to colour a specific feature by overriding the ``texture_format`` /
  ``cmap`` knobs (and optionally ``_clim()``). Same zero-copy contract as
  RasterPlot/VoltagePlot -- the value never leaves the GPU (see
  [[gpu-only-rendering]]).

  Color limits default to the feature's declared ``range`` (e.g. a Weight built
  with ``range=[-1, 1]``); when that range is non-finite (the default Weight
  range is ``[-inf, +inf]``) we fall back to a symmetric range read once from the
  initial values. Pass ``clim=(lo, hi)`` to override.
  """

  # --- knobs a subclass overrides for its feature ---
  texture_format = np.float32     # GL texture dtype; must match the value tensor's dtype
  cmap = 'viridis'                # vispy colormap name
  x_label = "Target neuron"
  y_label = "Source neuron"

  def __init__(self, source: str, target: str, feature_name: str,
               clim: tuple[float, float] | None = None, refresh_every: int = 1):
    super().__init__()            # heatmap: both axes linked to the panzoom camera
    self.source = source          # source layer name (connection key part 1)
    self.target = target          # target layer name (connection key part 2)
    self.feature_name = feature_name
    self._clim_override = clim    # explicit color limits; else range/data (see _clim)
    self.refresh_every = max(1, refresh_every)  # throttle big-matrix re-uploads
    self.connection = None        # Initialized in prime()
    self.feature = None
    self.visual = None
    self.colorbar = None

  def prime(self, network):
    self.connection = network.connections[(self.source, self.target)]
    self.feature = self.connection.feature_index[self.feature_name]

    value = self.feature.value
    if not isinstance(value, torch.Tensor) or value.is_sparse or value.dim() != 2:
      raise NotImplementedError(
        "FeaturePlot only supports dense 2D feature values (source.n x target.n); "
        f"got {type(value).__name__} shape={getattr(value, 'shape', None)} "
        f"sparse={getattr(value, 'is_sparse', None)}."
      )
    rows, cols = value.shape      # (source.n, target.n)
    clim = self._clim()

    self.visual = FeatureMatrix(
      rows=rows, cols=cols,
      value_getter=lambda: self.feature.value,  # re-fetch: value may be rebound
      texture_format=self.texture_format,
      clim=clim,
      cmap=self.cmap,
    )
    self.view.add(self.visual)
    self.view.camera.rect = (0, 0, cols, rows)
    self.y_axis.axis.axis_label = self.y_label
    self.x_axis.axis.axis_label = self.x_label
    self._add_colorbar(clim)

  def _add_colorbar(self, clim):
    # Vertical bar to the right of the view (grid col 2). White text/border: the
    # canvas bg is black and ColorBarWidget defaults to black. label_color also
    # colours the min/max tick labels (drawn from clim).
    self.colorbar = scene.ColorBarWidget(
      cmap=self.cmap, orientation='right',
      label=self.feature_name, clim=clim,
      label_color='white', border_color='white', border_width=1,
    )
    self.grid.add_widget(self.colorbar, row=0, col=2).width_max = 95

  def render(self, t):
    if t % self.refresh_every == 0:
      self.visual.migrate()

  def get_history(self):
    return None    # values live on the GPU; no CPU copy kept

  def _clim(self):
    # language=rst
    """Return ``(low, high)`` colour limits in feature-value units."""
    if self._clim_override is not None:
      return tuple(self._clim_override)

    # Prefer the feature's declared range, when it's a finite (lo < hi) scalar pair.
    rng = getattr(self.feature, "range", None)
    if rng is not None and len(rng) == 2:
      try:
        lo, hi = float(rng[0]), float(rng[1])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
          return (lo, hi)
      except (TypeError, ValueError):
        pass   # e.g. a non-scalar tensor range -> fall through to data-derived

    # Fallback: symmetric range from the initial values (a scalar reduction -- two
    # floats reach the host, not the matrix -- so the zero-copy render path is
    # untouched). Symmetric keeps 0 at the colormap center; rounded for a clean
    # colorbar label.
    m = self.feature.value.abs().max().item()
    if not np.isfinite(m) or m == 0.0:
      return (-1.0, 1.0)
    m = round(m, 3)
    return (-m, m)


class WeightPlot(FeaturePlot):
  # language=rst
  """
  Live heatmap of a :class:`Weight` feature's values. Uses a diverging colormap
  centered at 0 so excitatory (positive) and inhibitory (negative) weights read
  as opposite colours. Color limits follow the Weight's ``range`` if finite, else
  a symmetric range from the initial weights (see :class:`FeaturePlot`).
  """

  texture_format = np.float32
  cmap = 'coolwarm'              # diverging: low=blue, 0=white, high=red
