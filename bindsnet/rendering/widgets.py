from vispy import scene
import numpy as np
import torch
import colorsys
import types
import warnings

from abc import abstractmethod
from .visuals import (RasterHistory, VoltageHistory, FeatureMatrix, NeuronCloud,
                      SynapseLines, CachedSynapseLines, ScrollingLabels, ScrollingMarks)
from vispy.visuals.axis import Ticker
from vispy.scene.cameras import PanZoomCamera
from vispy.geometry import Rect
from vispy.color import get_colormap
from bindsnet.network.topology_features import Weight, Mask


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


def _sliding_update_subvisuals(self):
    # Drop-in for AxisVisual._update_subvisuals that skips rebuilding tick glyphs when
    # unchanged. The stock version reassigns text/anchors/label every domain change,
    # nulling TextVisual._vertices -> a full glyph re-layout next draw. A scrolling window
    # changes the domain every draw, so that ran every frame (~half steps/s). With a
    # FixedStepTicker the strings are stable, so cache them and touch the glyph-rebuilding
    # setters only on a real change; positions always re-upload (cheap).
    tick_pos, labels, tick_label_pos, anchors, axis_label_pos = self.ticker.get_update()
    # Axis line is static; re-upload its VBO only when it changes.
    if not np.array_equal(self.pos, self._cached_line_pos):
        self._line.set_data(pos=self.pos, color=self.axis_color)
        self._cached_line_pos = np.array(self.pos)
    self._ticks.set_data(pos=tick_pos, color=self.tick_color)

    labels = list(labels)
    if labels != self._cached_labels:
        self._text.text = labels            # strings changed -> rebuild glyphs
        self._cached_labels = labels
    # Only set anchors when they differ (a blind set nulls the vertex buffer too).
    if list(anchors) != list(self._text.anchors):
        self._text.anchors = anchors
    self._text.pos = tick_label_pos         # cheap: just slide label positions

    if self.axis_label is not None:
        if self.axis_label != self._cached_axis_label:
            self._axis_label_vis.text = self.axis_label
            self._cached_axis_label = self.axis_label
        self._axis_label_vis.pos = axis_label_pos
    self._need_update = False


def _make_axis_labels_slide(axis_widget):
    # Patch a linked AxisWidget so a sliding domain stops triggering per-draw glyph
    # re-layout (see _sliding_update_subvisuals). Instance-level method override.
    axis = axis_widget.axis
    # AxisVisual is Frozen; unfreeze to attach caches + the override, then re-freeze.
    axis.unfreeze()
    axis._cached_labels = None
    axis._cached_axis_label = None
    axis._cached_line_pos = None
    axis._update_subvisuals = types.MethodType(_sliding_update_subvisuals, axis)
    axis.freeze()


class AbstractWidget:
  _MARGIN = 10   # outer margin (px) around each widget's content

  def __init__(self, title: str | None = None):
    self.grid = scene.widgets.Grid(margin=self._MARGIN)
    # Explicit title; None -> a default from the component + widget type.
    self.title = title
    self.title_label = None     # created by subclasses that show a title

  def _default_title(self) -> str:
    # Auto title from component + widget type; base = class name.
    return type(self).__name__

  def _apply_title(self):
    # Explicit title if given, else the default. Called from prime().
    if self.title_label is not None:
      self.title_label.text = self.title if self.title is not None else self._default_title()

  @abstractmethod
  def prime(self, network, runtime):
    pass

  def capture(self, t):
    # Per-step GPU capture; runs EVERY step so throttled draws lose no data.
    # Default: none (the raster records spikes in network.step). Override for voltage.
    pass

  @abstractmethod
  def render(self, t):
    # Draw-time refresh (camera/axes/uniforms). Only on draw steps (draw_fps).
    pass

  def set_paused(self, paused: bool):
    # Play<->pause transition. Default: none (see GraphPlotWidget).
    pass

  def finish(self):
    # Sim complete. Default: none.
    pass

  def reset(self):
    # Sim reset to t=0. Default: none; override to restore the initial view.
    pass

  def reload(self, network):
    # Live model reload: the scaffolding (axes/gutter/colorbar/title) from prime() is
    # KEPT; only network-bound visuals + their GPU buffers are re-created via _bind().
    # Default: none (scaffold-only widgets). Requires prime() first.
    pass

  def chrome_nodes(self):
    # language=rst
    """
    Return ``(cached, live)`` scene-node lists for the chrome cache
    (:class:`~bindsnet.rendering.chrome_cache.ChromeCache`):

    * ``cached`` -- the widget's STATIC chrome (axis lines/ticks/labels, title). The
      cache bakes these into a texture once and hides them on normal frames, so vispy
      skips their costly per-draw CPU work; they reappear only during a re-bake.
    * ``live`` -- nodes that must be drawn live every frame (the data view, the
      scrolling gutter, the dynamic x-axis). The cache hides these only transiently
      while baking (so they aren't captured) and otherwise leaves their visibility to
      the widget; it never forces them on.

    Default: nothing cached (the widget renders normally).
    """
    return [], []

  def chrome_signature(self):
    # language=rst
    """
    Cheap, hashable snapshot of anything that changes the *cached* chrome pixels (an
    axis domain growing, a zoom/pan, ...). The cache re-bakes when it changes. Default:
    constant (never triggers a re-bake on its own).
    """
    return ()


# A plotting widget with x and y axes.
class GraphPlotWidget(AbstractWidget):
  # Columns the title spans (y-axis + view); spanning past them steals plot width.
  _title_col_span = 2

  def __init__(self, link_x=True, link_y=True, title: str | None = None):
    super().__init__(title=title)
    # Title centered over the plotting columns; text set in prime().
    self.title_label = scene.Label("", color='white', font_size=12, bold=True)
    self.title_label.height_max = 26
    self.grid.add_widget(self.title_label, row=0, col=0, col_span=self._title_col_span)

    self.y_axis = scene.AxisWidget(orientation='left', axis_label_margin=62)
    self.grid.add_widget(self.y_axis, row=1, col=0).width_max = 95

    self.view = self.grid.add_view(row=1, col=1, border_color='white')
    # Bounded camera, locked while running (render() drives the follow window; user zoom
    # would fight the per-draw reset). finish() unlocks it; limit_rect bounds zoom/pan.
    self.view.camera = BoundedPanZoomCamera()
    self.view.camera.interactive = False

    self.x_axis = scene.AxisWidget(orientation='bottom')
    self.grid.add_widget(self.x_axis, row=2, col=1).height_max = 55

    if link_x: self.x_axis.link_view(self.view)   # follows camera (inspect-mode labels)
    if link_y: self.y_axis.link_view(self.view)

    # The follow window slides the x-axis domain every draw; stop the per-frame glyph
    # re-layout that would otherwise ~halve steps/s. Harmless on the y-axis.
    _make_axis_labels_slide(self.x_axis)
    _make_axis_labels_slide(self.y_axis)

    self._initial_rect = None   # starting follow window, captured in each prime()

    ### Fixed-camera "oscilloscope" scrolling ###
    # Moving the camera every frame fires the transform cascade (re-layout both axes +
    # re-resolve every visual) -- ~halves steps/s. Instead PIN the camera and scroll the
    # data under it via one uniform (set_x_offset): no cascade. The vispy x-axis hides
    # while scrolling; the gutter axis below scrolls with the data. Opt in via
    # `_scroll_node` in prime.
    self._scroll_node = None    # Visual scrolled via set_x_offset; set in prime()
    self._scrolling = False     # in fixed-camera scroll mode
    self._abs_rect = None       # last trailing window in ABSOLUTE coords (for inspect)
    self._last_x0 = None        # last applied offset; skip redundant updates

    # Gutter ViewBox co-located with the vispy x-axis (same cell -> labels line up with
    # the data). Ticks + labels built once, scrolled by the same u_xoff. Shown only while
    # scrolling. None for non-scrolling widgets (heatmaps).
    self.x_scroll_view = None
    self.x_scroll_marks = None
    self.x_scroll_labels = None

  def _init_scroll(self, node):
    # From a subclass prime() once its scrolling visual exists.
    self._scroll_node = node
    self._scroll_node.set_x_offset(0)
    self._scrolling = False
    self._last_x0 = None
    self._abs_rect = self._initial_rect

  def _build_scroll_axis(self, step):
    # Build the gutter x-axis once: ticks + labels for the WHOLE timeline at constant
    # `step`. The gutter's x-range matches the plot's pinned window, so a label at time T
    # sits under data column T (both shifted left by u_xoff each draw).
    T = int(self.total_timesteps)
    W = float(self.window_size)
    step = float(step)

    # Pin the gutter to EXACTLY the vispy x-axis's pixel height (55) so the fraction-of-H
    # geometry below lands on the same pixels. min==max==H forces that height regardless
    # of layout. x is unconstrained: time -> width at any window size.
    H = 55.0
    self.x_scroll_view = scene.ViewBox(border_color=None)
    self.x_scroll_view.camera = PanZoomCamera(rect=Rect(0, 0, W, 1))
    self.x_scroll_view.camera.interactive = False
    cell = self.grid.add_widget(self.x_scroll_view, row=2, col=1)
    cell.height_min = cell.height_max = H

    majors = np.arange(0.0, T + 0.5 * step, step)
    mstep = step / 5.0          # 4 minor ticks per major, like FixedStepTicker
    minor = np.array([m + k * mstep for m in majors for k in (1, 2, 3, 4)
                      if m + k * mstep <= T], dtype=np.float32)

    # Match vispy AxisVisual's pixel metrics (baseline at top edge, major ticks 10 px,
    # minor 5 px, labels 22 px below), as fractions of H: 1 - px/H.
    major_y = 1.0 - 10.0 / H
    minor_y = 1.0 - 5.0 / H
    label_y = 1.0 - 22.0 / H
    white = (1.0, 1.0, 1.0, 1.0)   # vispy axis_color (baseline)
    grey = (0.7, 0.7, 0.7, 1.0)    # vispy tick_color (ticks)

    # GL_LINES in gutter data space; per-vertex colour draws baseline + ticks in one draw.
    segs = [[[0.0, 1.0], [float(T), 1.0]]]                          # axis baseline
    cols = [white, white]
    for m in majors:
      segs.append([[float(m), 1.0], [float(m), major_y]])          # major ticks
      cols += [grey, grey]
    for m in minor:
      segs.append([[float(m), 1.0], [float(m), minor_y]])          # minor ticks
      cols += [grey, grey]
    positions = np.array(segs, dtype=np.float32).reshape(-1, 2)
    colors = np.array(cols, dtype=np.float32)
    self.x_scroll_marks = ScrollingMarks(positions=positions, colors=colors)
    self.x_scroll_view.add(self.x_scroll_marks)

    # Number labels: one TextVisual for the whole timeline, scrolled by u_xoff.
    labels = ['%g' % m for m in majors]
    lpos = np.column_stack([majors, np.full(len(majors), label_y)]).astype(np.float32)
    self.x_scroll_labels = ScrollingLabels(
      text=labels, pos=lpos, color='white', font_size=8,
      anchor_x='center', anchor_y='top')
    self.x_scroll_view.add(self.x_scroll_labels)

    self.x_scroll_view.visible = False   # shown only while scrolling

  def _enter_scroll_mode(self):
    # Pin the camera and swap the gutter axis in for the vispy x-axis (hiding it stops
    # its per-draw tick/label upload; the gutter scrolls by one uniform).
    if self._scroll_node is None or self._scrolling:
      return
    self.view.camera.interactive = False
    cur = self.view.camera.rect
    self.view.camera.rect = (0, cur.bottom, self.window_size, cur.height)
    self.x_axis.visible = False
    self.x_scroll_view.visible = True
    self._scrolling = True
    self._last_x0 = None   # force the next _scroll_to

  def _exit_scroll_mode(self):
    # Back to ABSOLUTE coords + the dynamic vispy x-axis so paused/finished zoom-pan
    # relabels for any range.
    if self._scroll_node is None or not self._scrolling:
      return
    self._scroll_node.set_x_offset(0)
    self.x_scroll_marks.set_x_offset(0)
    self.x_scroll_labels.set_x_offset(0)
    self.x_scroll_view.visible = False
    self.x_axis.visible = True
    if self._abs_rect is not None:
      self.view.camera.rect = self._abs_rect   # reveal the same window, absolute coords
    self._scrolling = False

  def _scroll_to(self, x0):
    # Slide data + gutter so column x0 sits at the window's left edge. No camera move.
    # Skip when x0 is unchanged (the setters don't short-circuit on equal values).
    if not self._scrolling:
      self._enter_scroll_mode()
    if x0 == self._last_x0:
      return
    self._last_x0 = x0
    self._scroll_node.set_x_offset(x0)
    self.x_scroll_marks.set_x_offset(x0)
    self.x_scroll_labels.set_x_offset(x0)

  def reset(self):
    # Restore the starting follow window and re-lock the camera (reset may follow a
    # finish() that unlocked it).
    self._exit_scroll_mode()
    self._last_x0 = None
    if self._scroll_node is not None:
      self._scroll_node.set_x_offset(0)
      self.x_scroll_marks.set_x_offset(0)
      self.x_scroll_labels.set_x_offset(0)
    if self._initial_rect is not None:
      self._abs_rect = self._initial_rect
      self.view.camera.rect = self._initial_rect
    self.view.camera.interactive = False

  def set_paused(self, paused: bool):
    # Paused: hand the (bounded) camera to the user. Running: scroll under a pinned camera.
    if paused:
      self._exit_scroll_mode()
    self.view.camera.interactive = paused

  def finish(self):
    # Sim done: absolute coords, camera to the user (still bounded by limit_rect).
    self._exit_scroll_mode()
    self.view.camera.interactive = True

  def _detach_scroll_for_reload(self):
    # Reload: rewind the kept gutter to t=0 and show the dynamic x-axis. The gutter
    # geometry is reused (runtime/window_size don't change across reloads).
    if self.x_scroll_marks is not None:
      self.x_scroll_marks.set_x_offset(0)
      self.x_scroll_labels.set_x_offset(0)
      self.x_scroll_view.visible = False
    self.x_axis.visible = True
    self._scrolling = False
    self._last_x0 = None

  def chrome_nodes(self):
    # Bake the y-axis + title (static while scrolling). View + gutter are live. The vispy
    # x-axis is LIVE too: hidden while scrolling (gutter stands in), listing it here keeps
    # the bake from capturing it.
    live = [self.view, self.x_axis]
    if self.x_scroll_view is not None:
      live.append(self.x_scroll_view)
    return [self.y_axis, self.title_label], live

  def chrome_signature(self):
    # Only the y-axis is baked; its labels track the camera y-range. x is pinned.
    r = self.view.camera.rect
    return (round(float(r.bottom), 2), round(float(r.height), 2))


class VoltagePlot(GraphPlotWidget):
  # language=rst
  """
  Full-history, zero-copy voltage traces (x = absolute time, y = voltage).

  The voltage analogue of :class:`RasterPlot`. The layer's voltage is written in
  place by the node into a CUDA-registered GL buffer covering the WHOLE run (see
  :meth:`GUINetwork.enable_voltage_history`), and :class:`VoltageHistoryVisual`
  pulls ``v[t, neuron]`` via ``texelFetch`` in the vertex shader -- no per-frame
  copy, no ring buffer. During the run the camera follows a trailing window of
  ``window_size``; once the sim stops, zoom/pan freely to inspect all history
  back to t=0 (the x-axis is linked to the camera, so labels are real time).

  ``neuron_ids`` selects which of the layer's traces to draw; the full layer
  voltage is recorded (the node writes its whole ``v`` in place), so any neuron
  can be displayed.
  """

  def __init__(self,
               layer_name: str,
               neuron_ids: list[int],
               window_size: int = 100,
               y_range: tuple[float, float] = (-80.0, 40.0),
               title: str | None = None):
    super().__init__(title=title)   # x_axis linked to the camera: absolute-time labels
    self.layer_name = layer_name
    self.layer = None           # Initialized in prime()
    self.neuron_ids = list(neuron_ids)
    self.window_size = window_size   # trailing follow-window width
    self.total_timesteps = None # full history capacity (= runtime), set in prime()
    self.y_range = y_range      # initial / minimum y extent; grows to fit the data
    self.lines = None
    self._vmin = None           # GPU scalars: running observed voltage min/max
    self._vmax = None           # (updated in GUINetwork.step; read on draw)

  def _default_title(self) -> str:
    return f"{self.layer_name} — Voltage"

  def _bind(self, network):
    # Shared by prime() and reload(): alloc the voltage-history GL buffer, build the trace
    # visual, fit the camera. Ids past the (possibly smaller) layer size are dropped.
    self.layer = network.layers[self.layer_name]
    draw_ids = [i for i in self.neuron_ids if 0 <= i < int(self.layer.n)]

    info = network.enable_voltage_history(self.layer_name, self._runtime)
    self.total_timesteps = info['T']
    self._vmin, self._vmax = info['vmin'], info['vmax']   # in-place-updated scalars

    K = len(draw_ids)
    colors = np.array(
      [[*colorsys.hsv_to_rgb(i / max(K, 1), 0.9, 1.0), 1.0] for i in range(K)],
      dtype=np.float32)

    self.lines = VoltageHistory(
        neuron_ids=draw_ids, total_timesteps=info['T'],
        row_stride=info['row'], gl_buffer_id=info['vbo'], colors=colors,
    )
    self.view.add(self.lines)

    # Bound zoom/pan to the full extent; start on the first trailing window (y grows later).
    y0, y1 = self.y_range
    self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
    self._initial_rect = (0, y0, self.window_size, y1 - y0)
    self.view.camera.rect = self._initial_rect
    self._init_scroll(self.lines)   # scroll under a pinned camera
    self._last_y_extent = None      # last y-extent pushed; skip if unchanged

  def prime(self, network, runtime=None):
    if runtime is None:
      raise ValueError(
        "VoltagePlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")
    self._runtime = runtime
    self._bind(network)
    self.y_axis.axis.axis_label = "Voltage (mV)"
    self.x_axis.axis.axis_label = "Timestep"
    # Constant-step ticks so a sliding domain translates smoothly (no "nice" relocation).
    step = max(1, round(self.window_size / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)
    self._build_scroll_axis(step)
    self._apply_title()

  def reload(self, network):
    # Keep axes/gutter/title; swap the trace visual + GPU buffer (and vmin/vmax scalars).
    self._detach_scroll_for_reload()
    if self.lines is not None:
      self.lines.parent = None
      self.lines.release()
      self.lines = None
    self._bind(network)

  def _y_extent(self):
    # y_range grown to fit the observed voltage min/max so traces never clip. .item() is
    # the only readback (2 floats). Quantized to a grid so per-step jitter doesn't nudge
    # the camera every frame (a nudge fires the transform cascade + y-axis re-layout).
    y0, y1 = self.y_range
    vmin, vmax = self._vmin.item(), self._vmax.item()
    if np.isfinite(vmin) and np.isfinite(vmax):
      pad = 0.02 * max(1.0, vmax - vmin)        # keep extremes off the border
      y0, y1 = min(y0, vmin - pad), max(y1, vmax + pad)
    q = 5.0
    return float(np.floor(y0 / q) * q), float(np.ceil(y1 / q) * q)

  def render(self, t):
    # Scroll traces + gutter under a pinned camera. x stays pinned; y tracks the voltage
    # range only when the quantized extent changes. See _y_extent.
    x0 = max(0, t - self.window_size + 1)
    self._scroll_to(x0)
    y0, y1 = self._y_extent()
    if (y0, y1) != self._last_y_extent:
      self._last_y_extent = (y0, y1)
      self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
      self.view.camera.rect = (0, y0, self.window_size, y1 - y0)
    self._abs_rect = (x0, y0, self.window_size, y1 - y0)

  def reset(self):
    # Extremes are cleared on reset; forget the last y-extent so render() re-fits from t=0.
    self._last_y_extent = None
    super().reset()

  def finish(self):
    # Refresh bounds to the final extent (the last draw may predate it), then unlock.
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
  ``window_size``; once the sim stops, zoom/pan freely to inspect all history
  back to t=0 (the x-axis is linked to the camera, so labels are real time).
  """

  def __init__(self,
               layer_name: str,
               window_size: int = 100,
               title: str | None = None):
    super().__init__(title=title)   # x_axis linked to the camera: absolute-time labels
    self.layer_name = layer_name
    self.layer = None           # Initialized in prime()
    self.window_size = window_size   # trailing follow-window width
    self.total_timesteps = None # full history capacity (= runtime), set in prime()
    self.raster = None

  def _default_title(self) -> str:
    return f"{self.layer_name} — Raster"

  def _bind(self, network):
    # Shared by prime() and reload(): alloc the spike-history GL buffer, build the visual,
    # fit the camera to the (possibly new) layer size.
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    info = network.enable_spike_history(self.layer_name, self._runtime)
    self.total_timesteps = info['T']
    self.raster = RasterHistory(
        n_neurons=info['n'], total_timesteps=info['T'],
        row_stride=info['row'], gl_buffer_id=info['vbo'],
    )
    self.view.add(self.raster)
    # Bound zoom/pan to the full extent; start on the first trailing window.
    self.view.camera.limit_rect = Rect(0, 0, self.total_timesteps, self.layer_size)
    self._initial_rect = (0, 0, self.window_size, self.layer_size)
    self.view.camera.rect = self._initial_rect
    self._init_scroll(self.raster)   # scroll under a pinned camera

  def prime(self, network, runtime=None):
    if runtime is None:
      raise ValueError(
        "RasterPlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")
    self._runtime = runtime
    self._bind(network)
    self.y_axis.axis.axis_label = "Neuron"
    self.x_axis.axis.axis_label = "Timestep"
    # Constant-step ticks so a sliding domain translates smoothly.
    step = max(1, round(self.window_size / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)
    self._build_scroll_axis(step)
    self._apply_title()

  def reload(self, network):
    # Keep axes/gutter/title; swap the data visual + GPU buffer (layer size may differ).
    self._detach_scroll_for_reload()
    if self.raster is not None:
      self.raster.parent = None
      self.raster.release()
      self.raster = None
    self._bind(network)

  def render(self, t):
    # Scroll the raster under a pinned camera; _scroll_to slides the data and relabels the
    # detached x-axis. The camera never moves.
    x0 = max(0, t - self.window_size + 1)
    self._scroll_to(x0)
    self._abs_rect = (x0, 0, self.window_size, self.layer_size)


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

  _title_col_span = 3   # y-axis + view + colorbar

  #### Feature knobs (subclass overrides) ####
  texture_format = np.float32     # GL texture dtype; must match the value tensor's dtype
  cmap = 'viridis'                # vispy colormap name
  x_label = "Target neuron"
  y_label = "Source neuron"

  def __init__(self, source: str, target: str, feature_name: str,
               clim: tuple[float, float] | None = None, refresh_every: int = 1,
               title: str | None = None):
    super().__init__(title=title)
    self.source = source          # connection key part 1
    self.target = target          # connection key part 2
    self.feature_name = feature_name
    self._clim_override = clim    # explicit color limits; else range/data (see _clim)
    self.refresh_every = max(1, refresh_every)  # throttle big-matrix re-uploads
    self.connection = None        # set in prime()
    self.feature = None
    self.visual = None
    self.colorbar = None

  def _default_title(self) -> str:
    return f"{self.source} → {self.target} — {self.feature_name}"

  def _bind(self, network):
    # Shared by prime() and reload(): re-locate the connection/feature, build the heatmap
    # over its value matrix, fit the camera. Returns the colour limits for the colorbar.
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
    self.view.camera.limit_rect = Rect(0, 0, cols, rows)   # bound to the matrix extent
    self._initial_rect = (0, 0, cols, rows)
    self.view.camera.rect = self._initial_rect
    self.view.camera.interactive = True   # no follow window -> free (bounded) zoom/pan
    return clim

  def prime(self, network, runtime=None):
    clim = self._bind(network)
    self.y_axis.axis.axis_label = self.y_label
    self.x_axis.axis.axis_label = self.x_label
    self._add_colorbar(clim)
    self._apply_title()

  def reload(self, network):
    # Keep axes/colorbar/title; swap the heatmap for one bound to the new matrix.
    if self.visual is not None:
      self.visual.parent = None
      self.visual.release()
      self.visual = None
    clim = self._bind(network)
    # Refresh the colorbar legend if the range changed (best-effort, cosmetic).
    if self.colorbar is not None:
      try:
        self.colorbar.clim = clim
      except Exception:
        pass

  def _add_colorbar(self, clim):
    # Vertical bar right of the view (col 2). White text/border over the black canvas.
    self.colorbar = scene.ColorBarWidget(
      cmap=self.cmap, orientation='right',
      label=self.feature_name, clim=clim,
      label_color='white', border_color='white', border_width=1,
    )
    self.grid.add_widget(self.colorbar, row=1, col=2).width_max = 95

  def set_paused(self, paused: bool):
    pass   # camera stays interactive the whole run (no follow window)

  def reset(self):
    # Restore the full-matrix view and re-show the (cleared) values.
    if self._initial_rect is not None:
      self.view.camera.rect = self._initial_rect
    self.visual.migrate()

  def render(self, t):
    if t % self.refresh_every == 0:
      self.visual.migrate()

  def chrome_nodes(self):
    # Bake both axes + title; view is live. The colorbar is LIVE too -- its gradient
    # doesn't bake with usable alpha -- but it's static and cheap.
    return [self.y_axis, self.x_axis, self.title_label], [self.view, self.colorbar]

  def chrome_signature(self):
    # Both axes baked, camera interactive -> labels track the full rect; re-bake on zoom.
    r = self.view.camera.rect
    return (round(float(r.left), 2), round(float(r.bottom), 2),
            round(float(r.width), 2), round(float(r.height), 2))

  def _clim(self):
    # language=rst
    """Return ``(low, high)`` colour limits in feature-value units."""
    if self._clim_override is not None:
      return tuple(self._clim_override)

    # Prefer the feature's declared range, when a finite (lo < hi) scalar pair.
    rng = getattr(self.feature, "range", None)
    if rng is not None and len(rng) == 2:
      try:
        lo, hi = float(rng[0]), float(rng[1])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
          return (lo, hi)
      except (TypeError, ValueError):
        pass   # non-scalar tensor range -> data-derived

    # Fallback: symmetric range from the initial values (a scalar reduction, keeps 0
    # centred). Rounded for a clean colorbar label.
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


class NetworkPlot(AbstractWidget):
  # language=rst
  """
  Renders the network *structure* as a node-link diagram: neurons as circles laid
  out in layered columns (one column per layer), synapses as lines between them,
  with firing shown live on each neuron.

  Neurons are drawn by one :class:`NeuronCloudVisual` per layer, each reading that
  layer's spikes zero-copy from the shared spike-history GL buffer (the same buffer
  a :class:`RasterPlot` on the layer would use; see
  :meth:`GUINetwork.enable_spike_history`). A spiking neuron lights up and fades over
  a short ``afterglow`` window so it stays visible across throttled draws.

  Synapses are drawn by a single :class:`SynapseLinesVisual`. Connections are
  selected once at :meth:`prime` (host-side, off the render path): each masked
  synapse whose ``|weight| >= weight_threshold`` is a candidate, and if a connection
  has more than ``max_lines`` candidates the strongest by ``|weight|`` are kept (the
  cap is reported). Lines are coloured by weight on a diverging map. Recurrent / back
  edges (target column <= source column) bow outward so they read apart from the
  forward fan-out. The line-colour buffer is rebuildable
  (:meth:`SynapseLinesVisual.set_colors`) -- the hook for later weight-change views.

  Performance: the synapse lines are one GL_LINES draw whose cost is ~linear in the
  total vertex count (profiled at ~24 ms/frame for ~50k segments next to other
  plots). Two knobs bound it: ``max_lines`` caps each connection, and back-edge curve
  resolution is scaled down per-connection (``curve_segments`` -> straight) once the
  edge count is high, since the dominant cost is curved edges. See ``_CURVE_VERT_BUDGET``.
  """

  # Layout constants (data-space units; the camera auto-fits the result).
  _ROW_ASPECT = 4.0       # block is ~this many times taller than wide
  _SPACING = 1.0          # neuron-to-neuron grid spacing within a layer block

  # Curve-vertex budget per back-edge connection: curved edges (N x `curve_segments`
  # verts) dominate the vertex-bound line draw, so a connection's curve resolution is
  # scaled down past this, collapsing to straight when there are very many edges.
  _CURVE_VERT_BUDGET = 6000

  def __init__(self,
               layers: list[str] | None = None,
               connections: list[tuple[str, str]] | None = None,
               max_lines: int = 4_000,
               weight_threshold: float = 0.0,
               afterglow: int = 8,
               point_size: float = 9.0,
               line_alpha: float = 0.25,
               curve_segments: int = 6,
               title: str | None = None):
    super().__init__(title=title)
    self.layer_names = layers                 # None -> all layers (resolved in prime)
    self.connection_keys = connections        # None -> all connections
    self.max_lines = int(max_lines)           # per-connection synapse-line cap
    self.weight_threshold = float(weight_threshold)
    self.afterglow = int(afterglow)
    self.point_size = float(point_size)
    self.line_alpha = float(line_alpha)
    self.curve_segments = max(1, int(curve_segments))   # back-edge bow resolution

    self.title_label = scene.Label("", color='white', font_size=12, bold=True)
    self.title_label.height_max = 26
    self.grid.add_widget(self.title_label, row=0, col=0)

    self.view = self.grid.add_view(row=1, col=0, border_color='white')
    self.view.camera = BoundedPanZoomCamera(aspect=1)   # keep circles round
    self.view.camera.interactive = True

    self._positions = {}      # layer name -> (n, 2) float32 layout positions
    self._col = {}            # layer name -> column index (for back-edge detection)
    self.clouds = []          # NeuronCloud visuals (one per layer)
    self.synapses = None      # single SynapseLines visual
    self._initial_rect = None

  def _default_title(self) -> str:
    return "Network"

  #### Layout ####
  def _layout(self, network):
    # Each layer is a vertically-centered grid block, columns left->right in insertion
    # order. Block gap scales with the tallest block so columns stay distinct.
    names = self.layer_names or list(network.layers.keys())
    blocks = {}
    max_h = 1.0
    for name in names:
      n = int(network.layers[name].n)
      gc = max(1, int(np.ceil(np.sqrt(n / self._ROW_ASPECT))))   # grid columns
      gr = int(np.ceil(n / gc))                                  # grid rows
      blocks[name] = (n, gc, gr)
      max_h = max(max_h, (gr - 1) * self._SPACING)

    gap = max(4.0 * self._SPACING, 0.5 * max_h)
    x_cursor = 0.0
    for col, name in enumerate(names):
      n, gc, gr = blocks[name]
      idx = np.arange(n)
      cx = (idx % gc).astype(np.float32) * self._SPACING
      cy = (idx // gc).astype(np.float32) * self._SPACING
      cy -= (gr - 1) * self._SPACING / 2.0                       # center vertically
      pos = np.stack([x_cursor + cx, cy], axis=1).astype(np.float32)
      self._positions[name] = pos
      self._col[name] = col
      x_cursor += (gc - 1) * self._SPACING + gap                 # next column
    return names

  #### Synapse selection / geometry ####
  @staticmethod
  def _find_features(connection):
    weight = mask = None
    for f in connection.pipeline:
      if weight is None and isinstance(f, Weight):
        weight = f
      elif mask is None and isinstance(f, Mask):
        mask = f
    return weight, mask

  def _select_synapses(self, connection):
    # Device-side selection; only the capped index/weight set crosses to the host.
    # Returns (src_i, tgt_j, w) numpy arrays.
    weight, mask = self._find_features(connection)
    if weight is None:
      return None
    W = weight.value
    if not isinstance(W, torch.Tensor) or W.is_sparse or W.dim() != 2:
      warnings.warn(f"NetworkPlot: skipping non-dense-2D weight {weight.name}.")
      return None

    cand = W.abs() >= self.weight_threshold
    if mask is not None and isinstance(mask.value, torch.Tensor):
      cand = cand & mask.value.bool()
    idx = cand.nonzero(as_tuple=False)              # (K, 2): [src_i, tgt_j]
    K = int(idx.shape[0])
    if K == 0:
      return None
    wvals = W[idx[:, 0], idx[:, 1]]
    if K > self.max_lines:
      keep = torch.topk(wvals.abs(), self.max_lines).indices
      idx, wvals = idx[keep], wvals[keep]
      warnings.warn(
        f"NetworkPlot: connection {weight.name} has {K} synapses; drawing the "
        f"{self.max_lines} strongest by |weight| (raise max_lines to draw more).")
    return (idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy(),
            wvals.detach().to(torch.float32).cpu().numpy())

  @staticmethod
  def _curve(p0, p2, segments):
    # Quadratic-bezier polyline bowed perpendicular to each chord, as GL_LINES pairs.
    # p0/p2: (K, 2) -> verts (K*segments*2, 2).
    mid = 0.5 * (p0 + p2)
    d = p2 - p0
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
    norm = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.divide(perp, norm, out=np.zeros_like(perp), where=norm > 0)
    p1 = mid + perp * (0.25 * np.linalg.norm(d, axis=1, keepdims=True))
    ts = np.linspace(0.0, 1.0, segments + 1)
    a, b, c = (1 - ts) ** 2, 2 * (1 - ts) * ts, ts ** 2
    pts = (a[None, :, None] * p0[:, None, :]
           + b[None, :, None] * p1[:, None, :]
           + c[None, :, None] * p2[:, None, :])          # (K, S+1, 2)
    seg = np.stack([pts[:, :-1, :], pts[:, 1:, :]], axis=2)   # (K, S, 2, 2)
    return seg.reshape(-1, 2).astype(np.float32)

  @staticmethod
  def _straight(p0, p2):
    verts = np.empty((2 * len(p0), 2), dtype=np.float32)
    verts[0::2], verts[1::2] = p0, p2
    return verts

  def _build_synapses(self, network, names, bbox):
    keys = self.connection_keys or list(network.connections.keys())
    name_set = set(names)
    straight, straight_w, curved, curved_w = [], [], [], []
    for key in keys:
      src, tgt = key
      if src not in name_set or tgt not in name_set:
        continue
      sel = self._select_synapses(network.connections[key])
      if sel is None:
        continue
      i, j, w = sel
      p0 = self._positions[src][i]
      p2 = self._positions[tgt][j]
      back = self._col[tgt] <= self._col[src]        # recurrent edge -> bow
      # Curve resolution scaled to the edge count (fewer segments as it grows, straight
      # once a bow would be lost in the density). Forward edges are always straight.
      seg = min(self.curve_segments, max(1, self._CURVE_VERT_BUDGET // max(1, len(w)))) \
          if back else 1
      if seg <= 1:
        straight.append(self._straight(p0, p2))
        straight_w.append(np.repeat(w, 2))
      else:
        curved.append(self._curve(p0, p2, seg))
        curved_w.append(np.repeat(w, seg * 2))

    all_verts = straight + curved
    all_w = straight_w + curved_w
    if not all_verts:
      return
    verts = np.concatenate(all_verts, axis=0)
    wv = np.concatenate(all_w, axis=0)

    # Colour by weight on a diverging map, symmetric about 0 (clim from the weights).
    m = float(np.abs(wv).max()) if wv.size else 1.0
    m = m if (np.isfinite(m) and m > 0) else 1.0
    t = np.clip((wv + m) / (2 * m), 0.0, 1.0)
    colors = get_colormap('coolwarm').map(t).astype(np.float32)   # (V, 4)
    colors[:, 3] = self.line_alpha

    # Cached: the static lines bake once; the per-frame draw is a single quad.
    self.synapses = CachedSynapseLines(positions=verts, colors=colors, bbox=bbox)
    self.view.add(self.synapses)

  #### AbstractWidget API ####
  def _bind(self, network):
    # Shared by prime() and reload(): re-layout neurons, rebuild synapse lines + clouds,
    # fit the camera. The caller clears the visual lists first.
    names = self._layout(network)

    # Bounding box of all neuron positions (lines live within it).
    allpos = np.concatenate(list(self._positions.values()), axis=0)
    x0, y0 = allpos.min(axis=0)
    x1, y1 = allpos.max(axis=0)

    # Synapses first so neurons draw on top.
    self._build_synapses(network, names, (x0, y0, x1, y1))

    # One neuron cloud per layer, bound to that layer's shared spike history.
    K = len(names)
    for ci, name in enumerate(names):
      info = network.enable_spike_history(name, self._runtime)
      pos = self._positions[name]
      base = (*colorsys.hsv_to_rgb(ci / max(K, 1), 0.55, 0.85), 1.0)
      cloud = NeuronCloud(
        positions=pos, indices=np.arange(len(pos)),
        total_timesteps=info['T'], row_stride=info['row'], gl_buffer_id=info['vbo'],
        base_color=base, fire_color=(1.0, 0.95, 0.3, 1.0),
        point_size=self.point_size, glow=self.afterglow,
      )
      self.view.add(cloud)
      self.clouds.append(cloud)

    # Fit + bound the camera to the whole diagram, with a small margin.
    padx = 0.05 * max(1.0, x1 - x0)
    pady = 0.05 * max(1.0, y1 - y0)
    rect = (x0 - padx, y0 - pady, (x1 - x0) + 2 * padx, (y1 - y0) + 2 * pady)
    self.view.camera.limit_rect = Rect(*rect)
    self._initial_rect = rect
    self.view.camera.rect = rect

  def prime(self, network, runtime=None):
    if runtime is None:
      raise ValueError(
        "NetworkPlot needs the total runtime to size its spike-history buffers; "
        "it is supplied by Application.run().")
    self._runtime = runtime
    self._bind(network)
    self._apply_title()

  def reload(self, network):
    # Release the old clouds + cached lines, drop the stale layout, rebuild against the
    # new network.
    if self.synapses is not None:
      self.synapses.parent = None
      self.synapses.release()
      self.synapses = None
    for cloud in self.clouds:
      cloud.parent = None
      cloud.release()
    self.clouds = []
    self._positions = {}
    self._col = {}
    self._bind(network)

  def capture(self, t):
    pass   # spikes already in the GL history buffer (written in network.step)

  def render(self, t):
    # Re-bake the synapse texture only when stale (first frame / after set_colors).
    # Outside the scene draw so the nested FBO pass doesn't clobber the viewport.
    if self.synapses is not None and self.synapses.dirty:
      canvas = self.view.canvas
      if canvas is not None:
        canvas.set_current()
      self.synapses.refresh()
    for cloud in self.clouds:
      cloud.set_time(t)

  def reset(self):
    if self._initial_rect is not None:
      self.view.camera.rect = self._initial_rect
    for cloud in self.clouds:
      cloud.set_time(0)

  def chrome_nodes(self):
    # Only the static title is chrome; the live diagram view draws every frame.
    return [self.title_label], [self.view]
