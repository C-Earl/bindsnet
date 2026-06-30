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
    # Drop-in for vispy AxisVisual._update_subvisuals that avoids rebuilding the
    # tick-label glyphs when nothing about them changed. The stock version reassigns
    # `_text.text`, `_text.anchors` and the axis-label text on EVERY domain change;
    # each of those setters nulls TextVisual._vertices, forcing a full SDF glyph
    # re-layout (_text_to_vbo + new VertexBuffer) on the next draw. While a plot's
    # follow window scrolls, the domain changes every draw, so that re-layout ran
    # every frame and ~halved steps/s (see _make_axis_labels_slide). With a
    # FixedStepTicker the visible label strings are stable for many frames -- only
    # their positions slide -- so we cache strings/anchors and touch the
    # glyph-rebuilding setters only when they actually change. Positions are a cheap
    # attribute re-upload, so they are always refreshed.
    tick_pos, labels, tick_label_pos, anchors, axis_label_pos = self.ticker.get_update()
    # The axis line is static (self.pos only changes on resize), but the stock update
    # re-uploads its VBO every call -- pointless every draw while scrolling. Upload it
    # only when it actually changes.
    if not np.array_equal(self.pos, self._cached_line_pos):
        self._line.set_data(pos=self.pos, color=self.axis_color)
        self._cached_line_pos = np.array(self.pos)
    self._ticks.set_data(pos=tick_pos, color=self.tick_color)

    labels = list(labels)
    if labels != self._cached_labels:
        self._text.text = labels            # strings changed -> rebuild glyphs
        self._cached_labels = labels
    # The base Ticker returns a fresh-but-equal anchors list each call; assigning it
    # blindly would null the vertex buffer too, so only set it when it really differs.
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
    # Patch a linked AxisWidget's AxisVisual so a sliding domain stops triggering a
    # per-draw glyph re-layout (see _sliding_update_subvisuals). Instance-level
    # override: the bound method shadows the class method vispy calls from
    # _prepare_draw. The seed values guarantee the first update sets text/labels once.
    axis = axis_widget.axis
    # AxisVisual is a Frozen vispy object (rejects new attributes); open it to attach
    # our caches + the override, then re-freeze so typo-guarding still holds elsewhere.
    axis.unfreeze()
    axis._cached_labels = None
    axis._cached_axis_label = None
    axis._cached_line_pos = None
    axis._update_subvisuals = types.MethodType(_sliding_update_subvisuals, axis)
    axis.freeze()


class AbstractWidget:
  # Outer margin (px) around each widget's content, giving every plot a bit of
  # breathing room on all sides within its grid cell.
  _MARGIN = 10

  def __init__(self, title: str | None = None):
    # Grid to hold the widget title, view and axes (if applicable). The margin
    # pads the widget's content away from its cell edges on all sides.
    self.grid = scene.widgets.Grid(margin=self._MARGIN)
    # Explicit title; when None a default is generated from the plotted
    # component and the widget type (see _default_title / _apply_title).
    self.title = title
    self.title_label = None     # created by subclasses that show a title

  def _default_title(self) -> str:
    # Auto-generated title from the plotted component + widget type. Overridden
    # per widget; the base falls back to the class name.
    return type(self).__name__

  def _apply_title(self):
    # Set the title label text: the explicit `title` if given, else the
    # generated default. Called once the plotted component is known (prime()).
    if self.title_label is not None:
      self.title_label.text = self.title if self.title is not None else self._default_title()

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

  def reload(self, network):
    # Called by the app on a live model reload (Application.reload_model): the old
    # network has been replaced by a freshly built one, possibly with different layer
    # sizes. The one-time scene scaffolding (axes, scroll-axis gutter, colorbar, title)
    # built in prime() is KEPT; only the network-bound data visuals + their GPU history
    # buffers are released and re-created against `network` (the per-widget _bind()).
    # Default: nothing to do (scaffold-only widgets have no network binding). Requires
    # prime() to have run first (same total runtime / window size across reloads).
    pass


# A plotting widget with x and y axis
class GraphPlotWidget(AbstractWidget):
  # Number of columns the title spans = the widget's used columns (y-axis + view,
  # plus a colorbar column for FeaturePlot). Spanning past the used columns would
  # create an empty stretchy column that steals plot width.
  _title_col_span = 2

  def __init__(self, link_x=True, link_y=True, title: str | None = None):
    super().__init__(title=title)
    # Title centered across the plotting columns, above the axes/view. Text is
    # filled in by _apply_title() in prime(), once the component is known.
    self.title_label = scene.Label("", color='white', font_size=12, bold=True)
    self.title_label.height_max = 26
    self.grid.add_widget(self.title_label, row=0, col=0, col_span=self._title_col_span)

    self.y_axis = scene.AxisWidget(orientation='left', axis_label_margin=62)
    self.grid.add_widget(self.y_axis, row=1, col=0).width_max = 95

    self.view = self.grid.add_view(row=1, col=1, border_color='white')
    # Bounded camera, locked while the sim runs: render() drives a follow window
    # and we don't want user zoom fighting that per-draw reset (it makes the axes
    # glitch). finish() unlocks it once the run is over; limit_rect (set in each
    # subclass's prime) keeps zoom/pan inside the plotted data.
    self.view.camera = BoundedPanZoomCamera()
    self.view.camera.interactive = False

    self.x_axis = scene.AxisWidget(orientation='bottom')
    self.grid.add_widget(self.x_axis, row=2, col=1).height_max = 55

    if link_x: self.x_axis.link_view(self.view)   # follows camera (inspect-mode labels)
    if link_y: self.y_axis.link_view(self.view)

    # The follow window slides the x-axis domain every draw; stop that from
    # re-laying-out the tick-label glyphs each frame (it otherwise ~halves steps/s).
    # Harmless on the y-axis, whose domain rarely changes.
    _make_axis_labels_slide(self.x_axis)
    _make_axis_labels_slide(self.y_axis)

    self._initial_rect = None   # starting follow window, captured in each prime()

    # --- Fixed-camera "oscilloscope" scrolling -------------------------------
    # Moving the camera every frame to follow the newest data fires the vispy
    # transform cascade (recompute scene transform + re-layout BOTH linked axes +
    # re-resolve every visual) -- measured to ~halve steps/s the moment scrolling
    # begins (132 -> 89 on the 20k demo). Instead we PIN the camera and scroll the
    # data underneath it by writing a single shader uniform (set_x_offset): no camera
    # move, so NO cascade (frozen-camera diagnostic held 132 -> 132). The vispy x-axis
    # is hidden while scrolling and the gutter axis below scrolls with the data, so
    # labels stay on absolute time. Subclasses opt in by setting `_scroll_node` (the
    # visual to scroll) in prime.
    self._scroll_node = None    # the Visual we scroll via set_x_offset; set in prime()
    self._scrolling = False     # True while in fixed-camera scroll mode
    self._abs_rect = None       # last trailing window in ABSOLUTE coords (for inspect)
    self._last_x0 = None        # last applied scroll offset; skip redundant updates

    # Oscilloscope x-axis: a thin gutter ViewBox co-located with the vispy x-axis
    # (same grid cell -> same pixel x-extent, so labels line up with the data above).
    # Its ticks + labels are built once for the whole timeline and scrolled by the
    # same u_xoff uniform as the data (one scalar write/draw, no glyph re-layout).
    # Shown only while scrolling; on pause/finish we hide it and show the vispy
    # x-axis, whose dynamic ticker relabels for any zoom/pan during inspection.
    # None for non-scrolling widgets (heatmaps), which never build it.
    self.x_scroll_view = None
    self.x_scroll_marks = None
    self.x_scroll_labels = None

  def _init_scroll(self, node):
    # Called from a subclass prime() once its scrolling visual exists.
    self._scroll_node = node
    self._scroll_node.set_x_offset(0)
    self._scrolling = False
    self._last_x0 = None
    self._abs_rect = self._initial_rect

  def _build_scroll_axis(self, step):
    # Build the oscilloscope x-axis once: tick marks + labels for the WHOLE timeline
    # at constant `step`, in a gutter ViewBox co-located with the vispy x-axis. The
    # gutter camera's x range matches the plot's pinned window, so a label at absolute
    # time T sits exactly under data column T; both are shifted left by the same
    # u_xoff each draw.
    T = int(self.total_timesteps)
    W = float(self.window_size)
    step = float(step)

    # The gutter is pinned to EXACTLY the same pixel height as the vispy x-axis it
    # stands in for (height_max=55 in __init__), so the geometry below -- expressed
    # as fractions of that height -- lands on the same pixels the stock AxisVisual
    # would draw. Forcing min==max==H makes the row that height regardless of layout
    # pressure, so the fractions stay exact (and the camera maps data-y in [0,1] to
    # this height with no margin -- an explicitly-set PanZoomCamera.rect fills the
    # viewbox). x is unconstrained: the camera maps time -> width at any window size.
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

    # Match vispy AxisVisual's exact pixel metrics so the gutter is indistinguishable
    # from the stock x-axis when scroll<->pause swaps them: line at the TOP edge
    # (data-y=1, adjacent to the plot), major ticks hang 10 px down, minor ticks 5 px,
    # number labels anchored (center, top) at major_tick_length + tick_label_margin =
    # 10 + 12 = 22 px below the line. As fractions of the H px gutter: 1 - px/H.
    major_y = 1.0 - 10.0 / H
    minor_y = 1.0 - 5.0 / H
    label_y = 1.0 - 22.0 / H
    white = (1.0, 1.0, 1.0, 1.0)   # vispy axis_color (the baseline)
    grey = (0.7, 0.7, 0.7, 1.0)    # vispy tick_color (the ticks)

    # GL_LINES in gutter data space. The baseline spans the whole timeline so it
    # always covers the window after the u_xoff shift. Per-vertex colour: white
    # baseline + grey ticks in one draw (see ScrollingMarksVisual).
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
    # Pin the camera (x at 0, current y kept) and swap the gutter axis in for the
    # vispy x-axis: hiding the vispy axis stops its per-draw tick/label upload, and
    # the gutter scrolls by a single uniform instead.
    if self._scroll_node is None or self._scrolling:
      return
    self.view.camera.interactive = False
    cur = self.view.camera.rect
    self.view.camera.rect = (0, cur.bottom, self.window_size, cur.height)
    self.x_axis.visible = False
    self.x_scroll_view.visible = True
    self._scrolling = True
    self._last_x0 = None   # force the next _scroll_to to apply

  def _exit_scroll_mode(self):
    # Flip back to ABSOLUTE coords + the dynamic vispy x-axis so paused/finished
    # zoom-pan over the full history relabels for any range.
    if self._scroll_node is None or not self._scrolling:
      return
    self._scroll_node.set_x_offset(0)
    self.x_scroll_marks.set_x_offset(0)
    self.x_scroll_labels.set_x_offset(0)
    self.x_scroll_view.visible = False
    self.x_axis.visible = True                 # show the dynamic axis for inspection
    if self._abs_rect is not None:
      # Reveal the same window in absolute coords; with the x-axis now visible and
      # still camera-linked, this rect change relabels it to the absolute window.
      self.view.camera.rect = self._abs_rect
    self._scrolling = False

  def _scroll_to(self, x0):
    # Slide the data and the gutter axis so absolute column x0 sits at the left edge
    # of the pinned window. No camera move -> no transform cascade. Skip when x0 is
    # unchanged (e.g. the whole pre-scroll phase where x0 stays 0) -- the uniform
    # setters don't short-circuit on equal values.
    if not self._scrolling:
      self._enter_scroll_mode()
    if x0 == self._last_x0:
      return
    self._last_x0 = x0
    self._scroll_node.set_x_offset(x0)
    self.x_scroll_marks.set_x_offset(x0)       # one scalar write each -> ~free
    self.x_scroll_labels.set_x_offset(x0)

  def reset(self):
    # Restore the starting follow window and re-lock the camera; the sim advances
    # again from t=0 on the next Play/Step (set_paused re-locks too, but reset may
    # happen after finish() unlocked it).
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
    # While paused, hand the (bounded) camera to the user for zoom/pan over the full
    # absolute-time history; while running, scroll under a pinned camera. limit_rect
    # still keeps zoom/pan inside the plotted data.
    if paused:
      self._exit_scroll_mode()
    self.view.camera.interactive = paused

  def finish(self):
    # Sim done: flip to absolute coords and hand the camera to the user (still
    # bounded by limit_rect).
    self._exit_scroll_mode()
    self.view.camera.interactive = True

  def _detach_scroll_for_reload(self):
    # Live model reload: rewind the (kept) scroll-axis gutter to t=0 and show the
    # dynamic vispy x-axis, so the re-bound visual starts from a clean, unscrolled
    # state. The gutter geometry itself (built once in _build_scroll_axis) is reused --
    # runtime and window_size don't change across reloads, only the data buffer does.
    if self.x_scroll_marks is not None:
      self.x_scroll_marks.set_x_offset(0)
      self.x_scroll_labels.set_x_offset(0)
      self.x_scroll_view.visible = False
    self.x_axis.visible = True
    self._scrolling = False
    self._last_x0 = None


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
    # Network-dependent binding, shared by prime() (first run) and reload() (after a
    # live model rebuild). Allocates the voltage-history GL buffer on `network`, creates
    # the trace visual over it, and re-fits the camera. Neuron ids past the (possibly
    # new, smaller) layer size are dropped so texelFetch stays in range.
    self.layer = network.layers[self.layer_name]
    draw_ids = [i for i in self.neuron_ids if 0 <= i < int(self.layer.n)]

    # Allocate the GPU history buffer and route the layer's voltage into it.
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

    # Bound zoom/pan to the full plotted extent (all of time x the display range).
    y0, y1 = self.y_range
    self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
    # Start showing the first trailing window; absolute-time x means zoom-out
    # reveals all of history. y grows later as extremes appear (see render).
    self._initial_rect = (0, y0, self.window_size, y1 - y0)
    self.view.camera.rect = self._initial_rect
    # Scroll the traces under a pinned camera instead of panning the camera.
    self._init_scroll(self.lines)
    self._last_y_extent = None   # last (y0, y1) pushed to the camera; skip if unchanged

  def prime(self, network, runtime=None):
    if runtime is None:
      raise ValueError(
        "VoltagePlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")
    self._runtime = runtime
    self._bind(network)
    self.y_axis.axis.axis_label = "Voltage (mV)"
    self.x_axis.axis.axis_label = "Timestep"
    # Ticks at constant multiples of `step` so a sliding domain produces smoothly
    # translating labels instead of relocated "nice" ones (which jitter/swap).
    step = max(1, round(self.window_size / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)
    self._build_scroll_axis(step)
    self._apply_title()

  def reload(self, network):
    # Live model reload: keep the axes/scroll gutter/title; swap the trace visual + its
    # GPU buffer (and the running vmin/vmax scalars) for the new network.
    self._detach_scroll_for_reload()
    if self.lines is not None:
      self.lines.parent = None
      self.lines.release()
      self.lines = None
    self._bind(network)

  def _y_extent(self):
    # Dynamic y range: the configured y_range, grown outward to fit the observed
    # voltage min/max so traces never clip. .item() is a 2-float host sync (the
    # only readback); the values themselves never leave the GPU. Quantized to a grid
    # so tiny per-step jitter in the running min/max doesn't nudge the camera every
    # frame -- each nudge moves the camera, which fires the transform cascade and a
    # y-axis re-layout. With the running extremes (which saturate within a few steps)
    # this means the camera/y-axis update only on a genuine range change.
    y0, y1 = self.y_range
    vmin, vmax = self._vmin.item(), self._vmax.item()
    if np.isfinite(vmin) and np.isfinite(vmax):
      pad = 0.02 * max(1.0, vmax - vmin)        # keep extremes off the border
      y0, y1 = min(y0, vmin - pad), max(y1, vmax + pad)
    q = 5.0
    return float(np.floor(y0 / q) * q), float(np.ceil(y1 / q) * q)

  def render(self, t):
    # The shader reads the buffer live. Scroll the traces under a pinned camera via a
    # uniform (no transform cascade); the gutter x-axis scrolls the same way. The
    # camera's x stays pinned -- only y tracks the observed voltage range, and only
    # when the quantized extent actually changes (otherwise the camera never moves,
    # so neither it nor the y-axis does any work). See _y_extent.
    x0 = max(0, t - self.window_size + 1)
    self._scroll_to(x0)
    y0, y1 = self._y_extent()
    if (y0, y1) != self._last_y_extent:
      self._last_y_extent = (y0, y1)
      self.view.camera.limit_rect = Rect(0, y0, self.total_timesteps, y1 - y0)
      self.view.camera.rect = (0, y0, self.window_size, y1 - y0)
    self._abs_rect = (x0, y0, self.window_size, y1 - y0)

  def reset(self):
    # The running voltage extremes are cleared on reset; forget the last pushed
    # y-extent so render() re-fits the camera from t=0 instead of keeping the old
    # (saturated) range.
    self._last_y_extent = None
    super().reset()

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
    # Network-dependent binding, shared by prime() (first run) and reload() (after a
    # live model rebuild). Allocates the spike-history GL buffer on `network`, creates
    # the data visual over it, and fits the camera to the (possibly new) layer size.
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    info = network.enable_spike_history(self.layer_name, self._runtime)
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
    self._initial_rect = (0, 0, self.window_size, self.layer_size)
    self.view.camera.rect = self._initial_rect
    # Scroll the raster under a pinned camera instead of panning the camera.
    self._init_scroll(self.raster)

  def prime(self, network, runtime=None):
    if runtime is None:
      raise ValueError(
        "RasterPlot needs the total runtime to size its full-history buffer; "
        "it is supplied by Application.run().")
    self._runtime = runtime
    self._bind(network)
    self.y_axis.axis.axis_label = "Neuron"
    self.x_axis.axis.axis_label = "Timestep"
    # Ticks at constant multiples of `step` so a sliding domain produces smoothly
    # translating labels instead of relocated "nice" ones (which jitter/swap).
    step = max(1, round(self.window_size / 5 / 100) * 100) or 100
    self.x_axis.axis.ticker = FixedStepTicker(
      self.x_axis.axis, step=step, anchors=self.x_axis.axis.ticker._anchors)
    self._build_scroll_axis(step)
    self._apply_title()

  def reload(self, network):
    # Live model reload: keep the axes/scroll gutter/title, swap the data visual +
    # its GPU buffer for the new network (layer size may differ -> camera re-fits).
    self._detach_scroll_for_reload()
    if self.raster is not None:
      self.raster.parent = None
      self.raster.release()
      self.raster = None
    self._bind(network)

  def render(self, t):
    # The shader reads the buffer live. Scroll the raster under a pinned camera (no
    # transform cascade); _scroll_to slides the data and relabels the detached x-axis
    # to absolute time. The camera itself never moves while scrolling.
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

  # Three columns here: y-axis + view + colorbar, so the title spans all three.
  _title_col_span = 3

  # --- knobs a subclass overrides for its feature ---
  texture_format = np.float32     # GL texture dtype; must match the value tensor's dtype
  cmap = 'viridis'                # vispy colormap name
  x_label = "Target neuron"
  y_label = "Source neuron"

  def __init__(self, source: str, target: str, feature_name: str,
               clim: tuple[float, float] | None = None, refresh_every: int = 1,
               title: str | None = None):
    super().__init__(title=title)  # heatmap: both axes linked to the panzoom camera
    self.source = source          # source layer name (connection key part 1)
    self.target = target          # target layer name (connection key part 2)
    self.feature_name = feature_name
    self._clim_override = clim    # explicit color limits; else range/data (see _clim)
    self.refresh_every = max(1, refresh_every)  # throttle big-matrix re-uploads
    self.connection = None        # Initialized in prime()
    self.feature = None
    self.visual = None
    self.colorbar = None

  def _default_title(self) -> str:
    return f"{self.source} → {self.target} — {self.feature_name}"

  def _bind(self, network):
    # Network-dependent binding, shared by prime() (first run) and reload() (after a
    # live model rebuild). Re-locates the connection/feature on `network`, creates the
    # heatmap visual over its value matrix, and fits the camera. Returns the colour
    # limits so the caller can build/refresh the colorbar.
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
    return clim

  def prime(self, network, runtime=None):
    clim = self._bind(network)
    self.y_axis.axis.axis_label = self.y_label
    self.x_axis.axis.axis_label = self.x_label
    self._add_colorbar(clim)
    self._apply_title()

  def reload(self, network):
    # Live model reload: keep the axes/colorbar/title, swap the heatmap visual for one
    # bound to the new network's feature matrix (its shape may differ -> camera re-fits).
    if self.visual is not None:
      self.visual.parent = None
      self.visual.release()
      self.visual = None
    clim = self._bind(network)
    # Refresh the colorbar legend if the value range changed (best-effort: the stock
    # ColorBarWidget may not relabel, which is only cosmetic).
    if self.colorbar is not None:
      try:
        self.colorbar.clim = clim
      except Exception:
        pass

  def _add_colorbar(self, clim):
    # Vertical bar to the right of the view (grid col 2). White text/border: the
    # canvas bg is black and ColorBarWidget defaults to black. label_color also
    # colours the min/max tick labels (drawn from clim).
    self.colorbar = scene.ColorBarWidget(
      cmap=self.cmap, orientation='right',
      label=self.feature_name, clim=clim,
      label_color='white', border_color='white', border_width=1,
    )
    self.grid.add_widget(self.colorbar, row=1, col=2).width_max = 95

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

  # Synapse lines are redrawn every frame and the draw cost is ~linear in the total
  # vertex count (one GL_LINES draw, vertex-bound -- profiled: ~50k segments alongside
  # other plots costs ~24 ms/frame). Curved back-edges are the worst offender (N x
  # `curve_segments` vertices), so a back-edge connection's curve resolution is scaled
  # down once it would spend more than this many segment-vertices, collapsing to a
  # straight line when there are very many edges (where a smooth bow is invisible in
  # the hairball anyway).
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

    # Title across the top, with the diagram view below it.
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

  # --- layout ---------------------------------------------------------------
  def _layout(self, network):
    # Place each layer as a vertically-centered grid block, columns left->right in
    # the network's layer insertion order. Square grid spacing within a block; gap
    # between blocks scales with the tallest block so columns stay visually distinct.
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

  # --- synapse selection / geometry ----------------------------------------
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
    # Device-side selection (keeps the big matrices on the GPU); only the capped
    # index/weight set crosses to the host. Returns (src_i, tgt_j, w) numpy arrays.
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
    # Quadratic-bezier polyline bowed perpendicular to each chord, returned as
    # consecutive GL_LINES segment pairs. p0/p2: (K, 2). -> verts (K*segments*2, 2).
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
      # Curve resolution scaled to the edge count so the per-frame line-draw cost
      # stays bounded: full `curve_segments` for small connections, fewer as the
      # count grows, and a straight line (seg=1) once a smooth bow would be lost in
      # the density anyway. Forward edges are always straight (1 segment).
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

    # Colour by weight on a diverging map, symmetric about 0 so excitatory and
    # inhibitory read as opposite hues (clim from the selected weights).
    m = float(np.abs(wv).max()) if wv.size else 1.0
    m = m if (np.isfinite(m) and m > 0) else 1.0
    t = np.clip((wv + m) / (2 * m), 0.0, 1.0)
    colors = get_colormap('coolwarm').map(t).astype(np.float32)   # (V, 4)
    colors[:, 3] = self.line_alpha

    # Cached: bake the static lines into a texture once; the per-frame draw is then a
    # single camera-transformed quad (see CachedSynapseLinesVisual).
    self.synapses = CachedSynapseLines(positions=verts, colors=colors, bbox=bbox)
    self.view.add(self.synapses)

  # --- AbstractWidget API ---------------------------------------------------
  def _bind(self, network):
    # Network-dependent binding, shared by prime() (first run) and reload() (after a
    # live model rebuild). Re-lays-out the neurons, rebuilds the synapse lines + neuron
    # clouds against `network`, and fits the camera. Assumes the per-widget visual lists
    # were cleared by the caller (prime starts empty; reload releases the old ones).
    names = self._layout(network)

    # Data-space bounding box of all neuron positions (lines live within it).
    allpos = np.concatenate(list(self._positions.values()), axis=0)
    x0, y0 = allpos.min(axis=0)
    x1, y1 = allpos.max(axis=0)

    # Synapses first so neurons draw on top of the lines.
    self._build_synapses(network, names, (x0, y0, x1, y1))

    # One neuron cloud per layer, each bound to that layer's (shared) spike history.
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

    # Fit the camera to the whole diagram (with a small margin) and bound zoom/pan.
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
    # Live model reload: release the old clouds + cached synapse lines, drop the stale
    # layout, and rebuild everything against the new network (sizes/connectivity differ).
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
    # Spikes are already in the GL history buffer (written in network.step); nothing
    # to copy here.
    pass

  def render(self, t):
    # Re-bake the synapse texture only when stale (first frame / after set_colors);
    # the static lines are otherwise never redrawn -- the per-frame cost is one quad.
    # Done here (outside the scene draw) so the nested FBO pass doesn't clobber the
    # in-flight viewport; make the canvas context current first.
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
