from vispy import scene
import numpy as np
import torch
import colorsys
from abc import abstractmethod
from .visuals import RasterHistory, VoltageHistory, FeatureMatrix
from vispy.visuals.axis import Ticker
from vispy.scene.cameras import PanZoomCamera
from vispy.geometry import Rect


class BoundedPanZoomCamera(PanZoomCamera):
    """PanZoom camera that never zooms or pans past ``limit_rect`` -- the plotted
    data extent -- so the user can't scroll into the blank region around the data.

    The widget leaves the camera non-interactive while the sim runs (it drives a
    follow window itself via ``render``) and flips ``interactive`` True once the
    sim ends, so the completed history can be inspected freely but never beyond
    the data."""

    def __init__(self, *args, **kwargs):
        # Set before super().__init__: it assigns self.rect, which hits our setter.
        self.limit_rect = None     # Rect of the full data extent; set by the widget
        super().__init__(*args, **kwargs)

    @PanZoomCamera.rect.setter
    def rect(self, value):
        if isinstance(value, tuple):
            rect = Rect(*value)
        elif isinstance(value, Rect):
            rect = value
        else:
            rect = Rect(value)
        lim = self.limit_rect
        if lim is not None:
            w = min(rect.width, lim.width)        # never wider/taller than the data
            h = min(rect.height, lim.height)
            x = min(max(rect.left, lim.left), lim.right - w)     # keep inside bounds
            y = min(max(rect.bottom, lim.bottom), lim.top - h)
            rect = Rect(pos=(x, y), size=(w, h))
        PanZoomCamera.rect.fset(self, rect)


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

  def set_paused(self, paused: bool):
    # Called by the app on every play<->pause transition. Default: nothing to do.
    # Override for widgets that lock the camera while running so it can be handed
    # to the user for zoom/pan while paused (see GraphPlotWidget).
    pass

  def finish(self):
    # Called once when the simulation completes. Default: nothing to do. Override
    # for widgets that lock interaction during the run (graph plots).
    pass

  def reset(self):
    # Called by the app when the simulation is reset (network state cleared, time
    # back to 0). Default: nothing to do. Override to restore the initial camera
    # view; the underlying GL history buffers are cleared by GUINetwork.reset_history.
    pass


# A plotting widget with x and y axis
class GraphPlotWidget(AbstractWidget):
  def __init__(self, link_x=True, link_y=True):
    super().__init__()
    self.y_axis = scene.AxisWidget(orientation='left', axis_label_margin=62)
    self.grid.add_widget(self.y_axis, row=0, col=0).width_max = 95

    self.view = self.grid.add_view(row=0, col=1, border_color='white')
    # Bounded camera, locked while the sim runs: render() drives a follow window
    # and we don't want user zoom fighting that per-draw reset (it makes the axes
    # glitch). finish() unlocks it once the run is over; limit_rect (set in each
    # subclass's prime) keeps zoom/pan inside the plotted data.
    self.view.camera = BoundedPanZoomCamera()
    self.view.camera.interactive = False

    self.x_axis = scene.AxisWidget(orientation='bottom')
    self.grid.add_widget(self.x_axis, row=1, col=1).height_max = 55

    if link_x: self.x_axis.link_view(self.view)   # follows camera
    if link_y: self.y_axis.link_view(self.view)

    self._initial_rect = None   # starting follow window, captured in each prime()

  def reset(self):
    # Restore the starting follow window and re-lock the camera; the sim advances
    # again from t=0 on the next Play/Step (set_paused re-locks too, but reset may
    # happen after finish() unlocked it).
    if self._initial_rect is not None:
      self.view.camera.rect = self._initial_rect
    self.view.camera.interactive = False

  def set_paused(self, paused: bool):
    # While paused, hand the (bounded) camera to the user for zoom/pan; while
    # running, lock it so render()'s per-draw follow window isn't fought by user
    # input. limit_rect still keeps zoom/pan inside the plotted data.
    self.view.camera.interactive = paused

  def finish(self):
    # Sim done: hand the camera to the user (still bounded by limit_rect).
    self.view.camera.interactive = True


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
    self.y_range = y_range      # initial / minimum y extent; grows to fit the data
    self.lines = None
    self._vmin = None           # GPU scalars: running observed voltage min/max
    self._vmax = None           # (updated in GUINetwork.step; read on draw)

  def prime(self, network, runtime=None):
    self.layer = network.layers[self.layer_name]
    if runtime is None:
      raise ValueError(
        "VoltagePlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")

    # Allocate the GPU history buffer and route the layer's voltage into it.
    info = network.enable_voltage_history(self.layer_name, runtime)
    self.total_timesteps = info['T']
    self._vmin, self._vmax = info['vmin'], info['vmax']   # in-place-updated scalars

    K = len(self.neuron_ids)
    colors = np.array(
      [[*colorsys.hsv_to_rgb(i / max(K, 1), 0.9, 1.0), 1.0] for i in range(K)],
      dtype=np.float32)

    self.lines = VoltageHistory(
        neuron_ids=self.neuron_ids, total_timesteps=info['T'],
        row_stride=info['row'], gl_buffer_id=info['vbo'], colors=colors,
    )
    self.view.add(self.lines)

    # Bound zoom/pan to the full plotted extent (all of time x the display range).
    y0, y1 = self.y_range
    self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
    # Start showing the first trailing window; absolute-time x means zoom-out
    # reveals all of history. y grows later as extremes appear (see render).
    self._initial_rect = (0, y0, self.max_timesteps, y1 - y0)
    self.view.camera.rect = self._initial_rect
    self.y_axis.axis.axis_label = "Voltage (mV)"
    self.x_axis.axis.axis_label = "Timestep"
    # Ticks at constant multiples of `step` so a sliding domain produces smoothly
    # translating labels instead of relocated "nice" ones (which jitter/swap).
    step = max(1, round(self.max_timesteps / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)

  def _y_extent(self):
    # Dynamic y range: the configured y_range, grown outward to fit the observed
    # voltage min/max so traces never clip. .item() is a 2-float host sync (the
    # only readback); the values themselves never leave the GPU.
    y0, y1 = self.y_range
    vmin, vmax = self._vmin.item(), self._vmax.item()
    if np.isfinite(vmin) and np.isfinite(vmax):
      pad = 0.02 * max(1.0, vmax - vmin)        # keep extremes off the border
      y0, y1 = min(y0, vmin - pad), max(y1, vmax + pad)
    return y0, y1

  def render(self, t):
    # The shader reads the buffer live -- just slide the camera to follow the newest
    # activity; the linked x-axis relabels itself to absolute time. y expands to the
    # observed voltage range as the sim runs.
    x0 = max(0, t - self.max_timesteps + 1)
    y0, y1 = self._y_extent()
    self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
    self.view.camera.rect = (x0, y0, self.max_timesteps, y1 - y0)

  def finish(self):
    # Refresh bounds to the final observed extent (the last draw may predate the
    # final extreme), then hand the camera to the user.
    y0, y1 = self._y_extent()
    self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
    super().finish()


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
    # Bound zoom/pan to the full plotted extent (all of time x all neurons).
    self.view.camera.limit_rect = Rect(0, 0, self.total_timesteps, self.layer_size)
    # Start showing the first trailing window; absolute-time x means zoom-out
    # reveals all of history.
    self._initial_rect = (0, 0, self.max_timesteps, self.layer_size)
    self.view.camera.rect = self._initial_rect
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
    # Bound zoom/pan to the matrix extent (target x source neurons).
    self.view.camera.limit_rect = Rect(0, 0, cols, rows)
    self._initial_rect = (0, 0, cols, rows)
    self.view.camera.rect = self._initial_rect
    # Unlike the time-series plots, the heatmap has no follow window fighting the
    # user, so allow free (bounded) zoom/pan for the whole run.
    self.view.camera.interactive = True
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

  def set_paused(self, paused: bool):
    # The heatmap has no follow window fighting the user, so its camera is left
    # interactive for the whole run (see prime); pausing changes nothing here.
    pass

  def reset(self):
    # The heatmap camera stays interactive the whole run (no follow window); just
    # restore the initial full-matrix view and re-show the (now-cleared) values.
    if self._initial_rect is not None:
      self.view.camera.rect = self._initial_rect
    self.visual.migrate()

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
