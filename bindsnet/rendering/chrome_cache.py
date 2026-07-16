# language=rst
"""
Chrome caching for :class:`~bindsnet.rendering.app.Application`.

Profiling the GUI showed ``canvas._draw_scene`` is ~73% of every frame and that the
cost is almost entirely CPU-side: vispy re-processes each ``AxisWidget`` /
``TextVisual`` -- the plots' axis lines, ticks, labels, titles and the header (the
"chrome") -- on every draw, even though those pixels are identical frame to frame
(the GPU draws the whole scene in <3 ms; the CPU spends ~11 ms *issuing* it). The
zero-copy data visuals (rasters / voltage / heatmap) are cheap.

:class:`ChromeCache` bakes that static chrome into an off-screen texture ONCE and,
every frame, draws it back as a single fullscreen quad
(:class:`~bindsnet.rendering.visuals.ChromeOverlayVisual`) over the live data. The
expensive axis/text visuals are hidden during normal frames (so vispy skips them) and
shown only transiently during a re-bake. The cache re-bakes only when something that
affects the chrome actually changes -- a plot's axis domain (e.g. the voltage y-range
growing), the canvas size, or a model reload -- detected via a cheap per-widget
signature (see :meth:`AbstractWidget.chrome_signature`).

It runs only while the sim is advancing (the throughput-critical, fixed-camera
scrolling phase). While paused/finished it is disabled and the normal dynamic chrome
is shown, so zoom/pan inspection relabels the axes for any range.

Which nodes are baked vs drawn live is decided per widget by
:meth:`AbstractWidget.chrome_nodes`; the cache itself is widget-agnostic.
"""
import OpenGL.GL as gl
from vispy import gloo

from .visuals import ChromeOverlay


class ChromeCache:
  # language=rst
  """
  Bakes the static chrome of all of an Application's widgets into one texture and
  blits it each frame via a fullscreen overlay.

  :param canvas: the vispy ``SceneCanvas`` the plots render to.
  :param widgets: the Application's widgets (queried via ``chrome_nodes`` /
      ``chrome_signature``).
  :param header: optional canvas-level title ``Label`` (also static chrome).
  :param order: draw order of the overlay; high so it draws last (over the data).
  """

  def __init__(self, canvas, widgets, header=None, order=10):
    self.canvas = canvas
    self.widgets = widgets
    self.header = header
    self._tex = None
    self._fbo = None
    self._size = None              # (W, H) the fbo/texture is currently sized for
    self._active = False
    self._baked_sig = None         # signature the current texture was baked at
    self._cached_nodes = []        # static chrome: baked, hidden on live frames
    self._live_nodes = []          # data/gutter/dynamic: hidden only during a bake
    # Allocate overlay + fbo/texture now, not lazily on first bake, so the one-off GL
    # alloc never lands mid-run on a step doing CUDA<->GL interop.
    canvas.set_current()
    self._overlay = ChromeOverlay(parent=canvas.scene)
    self._overlay.order = order
    self._overlay.visible = False
    self._ensure_fbo()

  #### Node + signature gathering ####
  def _gather(self):
    # Static chrome (baked) vs live, per widget. Re-gathered on each enable() to pick
    # up late-added/reloaded widgets and current node identities.
    cached, live = [], []
    for w in self.widgets:
      c, l = w.chrome_nodes()
      cached += c
      live += l
    if self.header is not None:
      cached.append(self.header)
    self._cached_nodes = [n for n in cached if n is not None]
    self._live_nodes = [n for n in live if n is not None]

  def _signature(self):
    sig = [tuple(int(x) for x in self.canvas.physical_size)]
    for w in self.widgets:
      sig.append(w.chrome_signature())
    return tuple(sig)

  def _ensure_fbo(self):
    # (Re)allocate texture + framebuffer on canvas resize. The overlay quad is
    # clip-space, so only the texture resizes.
    W, H = (int(x) for x in self.canvas.physical_size)
    if self._size != (W, H):
      self._tex = gloo.Texture2D(shape=(H, W, 4), format='rgba',
                                 interpolation='linear')
      self._fbo = gloo.FrameBuffer(color=self._tex)
      self._overlay.set_texture(self._tex)
      self._size = (W, H)

  #### Lifecycle ####
  def enable(self):
    # Sim started advancing: gather nodes, arm a bake. NO GL work here (runs at the TOP
    # of a step, before network.step()) -- the bake is deferred to refresh() in the
    # DRAW phase, keeping the off-screen pass off the interop path. Until then the
    # normal chrome draws for at most one throttled frame.
    self._active = True
    self._gather()
    self._baked_sig = None

  def disable(self):
    # Paused/finished/reset: show dynamic chrome (relabels on zoom/pan), hide overlay.
    self._active = False
    self._overlay.visible = False
    for n in self._cached_nodes:
      n.visible = True

  def refresh(self):
    # Once per DRAW frame while running, before canvas.update(). Re-bakes only when the
    # signature changed; else reuses the baked texture for free.
    if not self._active:
      return
    sig = self._signature()
    if sig != self._baked_sig:
      try:
        self._rebake()
        self._baked_sig = sig
      except Exception:
        self.disable()   # a cache failure must never break rendering

  def _rebake(self):
    canvas = self.canvas
    canvas.set_current()
    self._ensure_fbo()
    # Bake: hide live nodes, show static chrome, hide the overlay, render into the fbo.
    # Live-node visibility is saved/restored so the widgets' scroll-mode bookkeeping
    # (gutter vs x-axis) is left untouched.
    saved = [(n, n.visible) for n in self._live_nodes]
    for n, _ in saved:
      n.visible = False
    for n in self._cached_nodes:
      n.visible = True
    self._overlay.visible = False
    try:
      # vispy's own offscreen path (push_fbo/_draw_scene/pop_fbo): it manages the canvas
      # fbo + viewport stacks and the scene TransformSystem. A raw gloo.FrameBuffer
      # diverged vispy's viewport stack from GL -> next CUDA<->GL map failed
      # (CUDA_ERROR_INVALID_GRAPHICS_CONTEXT). _draw_scene clears transparent so the
      # plot regions stay see-through.
      canvas.push_fbo(self._fbo, (0, 0), canvas.size)
      try:
        canvas._draw_scene(bgcolor=(0.0, 0.0, 0.0, 0.0))
      finally:
        canvas.pop_fbo()
      gl.glFinish()   # land the bake before next frame's blit (rare, so cheap)
    finally:
      # Always restore live nodes + current context, even if the bake raised.
      canvas.set_current()
      for n, ov in saved:
        n.visible = ov
    # Live: hide the baked static chrome, show the blit quad.
    for n in self._cached_nodes:
      n.visible = False
    self._overlay.visible = True

  def release(self):
    # Detach overlay, drop GL objects (best-effort; e.g. canvas teardown).
    if self._overlay is not None:
      try:
        self._overlay.parent = None
      except Exception:
        pass
    self._overlay = None
    self._fbo = None
    self._tex = None
