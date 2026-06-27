from vispy import scene
import numpy as np
import torch
import colorsys
from abc import abstractmethod
from .visuals import RasterHistory, VoltageHistory, FeatureMatrix
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
    self.grid = scene.widgets.Grid()    # Grid to hold widget view and axes (if applicable)

  @abstractmethod
  def prime(self, network, runtime):
    pass

  def capture(self, t):
    # Cheap per-step data capture into GPU buffers. Runs EVERY step regardless of
    # draw rate, so throttling the draw never drops data. Default: nothing to do
    # (e.g. the history raster records spikes in network.step). Override for
    # widgets that fill a ring buffer (voltage).
    pass

  @abstractmethod
  def render(self, t):
    # Draw-time refresh: update camera/axes/uniforms and request a redraw. Runs only
    # on draw steps (see Application.draw_fps throttling).
    pass


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
  # language=rst
  """
  Full-history, zero-copy voltage traces (x = absolute time, y = voltage).

  The voltage analogue of :class:`RasterPlot`. The layer's voltage is written in
  place by the node into a CUDA-registered GL buffer covering the WHOLE run (see
  :meth:`GUINetwork.enable_voltage_history`), and :class:`VoltageHistoryVisual`
  pulls ``v[t, neuron]`` via ``texelFetch`` in the vertex shader -- no per-frame
  copy, no ring buffer. During the run the camera follows a trailing window of
  ``max_timesteps``; once the sim stops, zoom/pan freely to inspect all history
  back to t=0 (the x-axis is linked to the camera, so labels are real time).

  ``neuron_ids`` selects which of the layer's traces to draw; the full layer
  voltage is recorded (the node writes its whole ``v`` in place), so any neuron
  can be displayed.
  """

  def __init__(self,
               layer_name: str,
               neuron_ids: list[int],
               max_timesteps: int = 100,
               y_range: tuple[float, float] = (-80.0, 40.0)):
    super().__init__()          # x_axis linked to the camera: absolute-time labels
    self.layer_name = layer_name
    self.layer = None           # Initialized in prime()
    self.neuron_ids = list(neuron_ids)
    self.max_timesteps = max_timesteps   # trailing follow-window width
    self.total_timesteps = None # full history capacity (= runtime), set in prime()
    self.y_range = y_range
    self.lines = None

  def prime(self, network, runtime=None):
    self.layer = network.layers[self.layer_name]
    if runtime is None:
      raise ValueError(
        "VoltagePlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")

    # Allocate the GPU history buffer and route the layer's voltage into it.
    info = network.enable_voltage_history(self.layer_name, runtime)
    self.total_timesteps = info['T']

    K = len(self.neuron_ids)
    colors = np.array(
      [[*colorsys.hsv_to_rgb(i / max(K, 1), 0.9, 1.0), 1.0] for i in range(K)],
      dtype=np.float32)

    self.lines = VoltageHistory(
        neuron_ids=self.neuron_ids, total_timesteps=info['T'],
        row_stride=info['row'], gl_buffer_id=info['vbo'], colors=colors,
    )
    self.view.add(self.lines)

    # Start showing the first trailing window; absolute-time x means zoom-out
    # reveals all of history.
    y0, y1 = self.y_range
    self.view.camera.rect = (0, y0, self.max_timesteps, y1 - y0)
    self.y_axis.axis.axis_label = "Voltage (mV)"
    self.x_axis.axis.axis_label = "Timestep"
    # Ticks at constant multiples of `step` so a sliding domain produces smoothly
    # translating labels instead of relocated "nice" ones (which jitter/swap).
    step = max(1, round(self.max_timesteps / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)

  def render(self, t):
    # The shader reads the buffer live -- just slide the camera to follow the newest
    # activity; the linked x-axis relabels itself to absolute time.
    x0 = max(0, t - self.max_timesteps + 1)
    y0, y1 = self.y_range
    self.view.camera.rect = (x0, y0, self.max_timesteps, y1 - y0)


class RasterPlot(GraphPlotWidget):
  # language=rst
  """
  Full-history, true-zero-copy spike raster (x = absolute time, y = neuron).

  The layer's spikes are written in place by the node into a CUDA-registered GL
  buffer covering the WHOLE run (see :meth:`GUINetwork.enable_spike_history`),
  and :class:`RasterHistoryVisual` reads it via ``texelFetch`` -- no per-frame
  copy, no ring buffer. During the run the camera follows a trailing window of
  ``max_timesteps``; once the sim stops, zoom/pan freely to inspect all history
  back to t=0 (the x-axis is linked to the camera, so labels are real time).
  """

  def __init__(self,
               layer_name: str,
               max_timesteps: int = 100):
    super().__init__()          # x_axis linked to the camera: absolute-time labels
    self.layer_name = layer_name
    self.layer = None           # Initialized in prime()
    self.max_timesteps = max_timesteps   # trailing follow-window width
    self.total_timesteps = None # full history capacity (= runtime), set in prime()
    self.raster = None

  def prime(self, network, runtime=None):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    if runtime is None:
      raise ValueError(
        "RasterPlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")

    # Allocate the GPU history buffer and route the layer's spikes into it.
    info = network.enable_spike_history(self.layer_name, runtime)
    self.total_timesteps = info['T']
    self.raster = RasterHistory(
        n_neurons=info['n'], total_timesteps=info['T'],
        row_stride=info['row'], gl_buffer_id=info['vbo'],
    )
    self.view.add(self.raster)
    # Start showing the first trailing window; absolute-time x means zoom-out
    # reveals all of history.
    self.view.camera.rect = (0, 0, self.max_timesteps, self.layer_size)
    self.y_axis.axis.axis_label = "Neuron"
    self.x_axis.axis.axis_label = "Timestep"
    # Ticks at constant multiples of `step` so a sliding domain produces smoothly
    # translating labels instead of relocated "nice" ones (which jitter/swap).
    step = max(1, round(self.max_timesteps / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)

  def render(self, t):
    # The shader reads the buffer live -- just slide the camera to follow the newest
    # activity; the linked x-axis relabels itself to absolute time.
    x0 = max(0, t - self.max_timesteps + 1)
    self.view.camera.rect = (x0, 0, self.max_timesteps, self.layer_size)


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

  def prime(self, network, runtime=None):
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
