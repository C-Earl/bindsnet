from vispy import app, scene
import time
import torch
from bindsnet.rendering.widgets import AbstractWidget
from bindsnet.rendering.controls import QtControlPanel
from bindsnet.network.network import GUINetwork

# Plots always render on the GLFW backend. GLFW renders straight to the window (bare
# buffer swap) and drives the sim from a tight poll-loop, so the simulation runs at
# full speed. vispy's Qt backend instead renders via an offscreen QOpenGLWidget that
# Qt composites every frame, and dispatches each sim step through a QTimer + the Qt
# event loop -- both much slower (this is why embedding the canvas in Qt was sluggish
# even at 1:1 DPI). Controls live outside the render path; see `controls` below.
app.use_app('glfw')


class Application():
  def __init__(self, network: GUINetwork,
               width=1400, height=900, title="BindsNET GUI",
               header: str | None = None,
               max_steps_per_second: int | float | str | None = None,
               draw_fps: float | None = None,
               parameters: dict | None = None):
    self.width, self.height = width, height

    # `network` is a GUINetwork. If it OVERRIDES build() (an inheritable model
    # definition), the Application drives its lifecycle: build() assembles it,
    # make_input() supplies the stimulus, network.parameters feeds the editable panel,
    # and the "Apply & Reload" button rebuilds it in place (see reload_model). A plain
    # GUINetwork assembled imperatively (no build() override) is used as-is with no
    # reload, driven by the `inputs` dict passed to run().
    self.network = network
    self._buildable = type(network).build is not GUINetwork.build
    self._has_make_input = type(network).make_input is not GUINetwork.make_input
    self.can_reload = self._buildable
    if self._buildable:
      self.network.rebuild()                    # initial build() from constructor params
      self.parameters = dict(network.parameters)
    else:
      self.parameters = parameters              # legacy: cosmetic-only panel rows
    self.widgets = []
    self.inputs = None      # Set when run() is called; Inputs into network during runtime
    self.runtime = None     # Set when run() is called; Total runtime of network simulation
    self.current_time = 0   # Current timestep in network; incremented during runtime

    # Cap on how many sim steps run per wall-clock second. `inf` (or "inf"/"max")
    # means "go as fast as possible" -- a 0s timer interval, i.e. one step per tick
    # with no throttle. Defaults to the draw rate so, out of the box, the sim steps
    # roughly in lock-step with the redraws (fall back to 60 if neither is set).
    if max_steps_per_second is None:
      max_steps_per_second = draw_fps if draw_fps is not None else 60
    self.max_steps_per_second = self._coerce_sps(max_steps_per_second)

    # Decouple the (expensive) full-canvas redraw from the simulation rate: run the
    # sim + cheap per-step data capture (widget.capture) every step, but redraw
    # (widget.render + canvas redraw + swap) at most `draw_fps` times/second. No
    # data is lost because capture is independent of drawing. draw_fps=None draws
    # every step.
    self.draw_fps = draw_fps
    self._last_draw = None

    # --- Simulation run-state, driven by whichever control panel is active -----
    # The timer always ticks; whether a tick advances the sim depends on this state.
    #   running     : continuous play (Play/Pause)
    #   step_budget : count of discrete steps queued (Step / Run N), consumed one
    #                 per tick even while `running` is False.
    # "active" (advancing) == running or step_budget > 0; otherwise the sim idles
    # and the plot cameras are handed to the user for zoom/pan.
    self.running = False
    self.step_budget = 0
    self._was_active = None   # last active-state, to fire play<->pause transitions once

    # Rolling measurement of the ACTUAL steps/second, reported to the panel ~2x/sec.
    self._sps_count = 0       # steps taken since the last measurement window opened
    self._sps_t0 = None       # perf_counter at the start of the current window

    # Initialize VisPy canvas (GLFW) and grid layout for widget rendering.
    self.canvas = scene.SceneCanvas(
      title=title,
      keys='interactive',
      bgcolor='black',
      size=(self.width, self.height),
      show=True,
    )
    # Top-level layout: an optional centered title spanning the full width, above
    # the grid of plotting widgets. When there's no title we still reserve a small
    # spacer row so the topmost axis tick labels aren't clipped by the canvas edge.
    self.layout = self.canvas.central_widget.add_grid(margin=0)
    next_row = 0
    # Top padding above everything, so the header (or the topmost tick labels when
    # there's no header) isn't flush against the canvas edge.
    self.layout.add_widget(row=next_row, col=0).height_max = 24
    next_row += 1
    if header is not None:
      self.title_label = scene.Label(header, color='white', font_size=20, bold=True)
      self.title_label.height_max = 48
      self.layout.add_widget(self.title_label, row=next_row, col=0)
      next_row += 1
    else:
      self.title_label = None

    self.grid = self.layout.add_grid(row=next_row, col=0, margin=10)

    # Migrate network tensors to shared OpenGL buffers
    self.network.migrate()

    # Build the control surface (a separate Qt window). It calls back into
    # toggle_play/step_once/run_n and reads back via set_time etc.
    self.panel = QtControlPanel(self, parameters=self.parameters)

  # --- steps-per-second rate -------------------------------------------------
  @staticmethod
  def _coerce_sps(value: int | float | str) -> float:
    # Accept a number, or the words "inf"/"max"/"unlimited"/"" for "as fast as
    # possible". Returns a positive float (possibly math.inf).
    if isinstance(value, str):
      if value.strip().lower() in ("", "inf", "max", "unlimited"):
        return float("inf")
      value = float(value)
    value = float(value)
    if value <= 0:
      raise ValueError(f"max_steps_per_second must be > 0, got {value}")
    return value

  @staticmethod
  def _interval_for(sps: float) -> float:
    # inf steps/sec -> a 0s timer interval (vispy fires it as fast as it can).
    return 0.0 if sps == float("inf") else 1.0 / sps

  def set_max_steps_per_second(self, value: int | float | str):
    # Update the cap live; the running timer's interval is swapped in place.
    self.max_steps_per_second = self._coerce_sps(value)
    if hasattr(self, "timer"):
      self.timer.interval = self._interval_for(self.max_steps_per_second)

  def add_widget(self, widget: AbstractWidget, row: int, col: int):
    self.widgets.append(widget)
    self.grid.add_widget(widget.grid, row, col)
    # Priming is deferred to run(): some widgets (full-history raster) need the
    # total runtime to size their GPU buffers, and runtime isn't known until run().
    # Support adding widgets after run() too, in which case prime immediately.
    if self.runtime is not None:
      widget.prime(self.network, self.runtime)

  # --- Control callbacks (panel-agnostic) ------------------------------------
  def toggle_play(self):
    self.running = not self.running
    self.panel.set_playing(self.running)

  def step_once(self):
    # Queue a single simulation step; consumed on the next tick even while paused.
    self.step_budget += 1

  def run_n(self, n: int):
    # Queue N steps.
    if n and n > 0:
      self.step_budget += int(n)

  def reset(self):
    # Clear the network's live state AND its recorded GL history, rewind to t=0,
    # restore every plot's initial view, and re-arm the run-state machine. The timer
    # is restarted in case the run had already finished (step() stops it at the end).
    self.running = False
    self.step_budget = 0
    self._was_active = None
    self._last_draw = None
    self._sps_count = 0
    self._sps_t0 = None
    self.current_time = 0
    self.network.reset_state_variables()
    self.network.reset_history()
    for widget in self.widgets:
      widget.reset()
    # Re-lock cameras to the running (non-interactive) state via the transition path.
    self._set_active(False)
    if hasattr(self, "timer") and not self.timer.running:
      self.timer.start()
    self.panel.on_reset()
    self.canvas.update()

  def reload_model(self):
    # language=rst
    """
    Rebuild the network from the control panel's current parameters and re-bind the
    plots in place, WITHOUT recreating the canvas / control window (that is what causes
    the black-screen/lag we avoid). Driven by the panel's "Apply & Reload" button, which
    fires on the main GL thread during panel.pump() -- the same path reset() uses to
    touch GL, so the context is current and these calls are safe.

    The network's :meth:`GUINetwork.rebuild` reassembles the model IN PLACE (frees the
    old GL buffers, tears down the layers, re-runs build() with the edited parameters),
    so the same network object is kept and the widgets simply re-bind to it.
    """
    if not self.can_reload:
      return
    if self.runtime is None:
      self.panel.show_status("Start the run before reloading.", error=True)
      return

    # Coerce the edited fields first; a bad value aborts before the model is touched.
    try:
      values = self.panel.get_parameter_values()
    except Exception as exc:
      self.panel.show_status(f"Invalid parameter: {exc}", error=True)
      return

    self.canvas.set_current()   # ensure GL calls below target the plot context
    try:
      self.network.rebuild(**values)   # free GL, clear, set params, re-run build()
      self.network.migrate()           # allocate the rebuilt net's shared GL buffers
    except Exception as exc:
      # build() failed (e.g. an invalid size). The network is now an empty shell, but
      # rebuild() always clears-then-builds, so a subsequent reload with good values
      # fully recovers. Leave the sim paused and report.
      self.panel.show_status(f"Build failed: {exc}", error=True)
      return

    # Re-bind every plot to the rebuilt network (releases old visuals, allocates fresh
    # history buffers). Done before regenerating inputs so a (rare) input mismatch can't
    # leave widgets pointing at freed buffers.
    for widget in self.widgets:
      widget.reload(self.network)
    self.parameters = dict(self.network.parameters)

    # Regenerate the stimulus for the rebuilt model. make_input() tracks the parameters
    # so it always fits; a fixed dict is only warned about if it no longer matches.
    input_warning = None
    if self._has_make_input:
      self.inputs = self.network.make_input(self.runtime)
    else:
      try:
        self._validate_inputs(self.inputs, self.network)
      except Exception as exc:
        input_warning = f"Reloaded, but inputs no longer fit: {exc}"

    # Re-arm the run-state machine at t=0 (the rebuilt net + its history buffers are
    # already fresh/zeroed, so no reset_state_variables/reset_history needed).
    self.running = False
    self.step_budget = 0
    self._was_active = None
    self._last_draw = None
    self._sps_count = 0
    self._sps_t0 = None
    self.current_time = 0
    self._set_active(False)
    if hasattr(self, "timer") and not self.timer.running:
      self.timer.start()

    self.panel.on_reset()
    self.panel.set_parameter_values(self.network.parameters)
    if input_warning is not None:
      self.panel.show_status(input_warning, error=True)
    else:
      self.panel.show_status("Model reloaded.")
    self.canvas.update()

  def _validate_inputs(self, inputs: dict, network: GUINetwork):
    # Ensure the (possibly regenerated) inputs match the rebuilt network before swapping
    # anything in: each input tensor's last dim must equal its target layer's n, and a
    # time-major tensor must cover the whole runtime. Raises ValueError on a mismatch.
    for name, tensor in inputs.items():
      if name not in network.layers:
        continue
      n = int(network.layers[name].n)
      if int(tensor.shape[-1]) != n:
        raise ValueError(
          f"input '{name}' last dim {int(tensor.shape[-1])} != layer '{name}' n {n}. "
          f"Pass `inputs` to run() as a builder function so it tracks the parameters.")
      if tensor.dim() >= 2 and int(tensor.shape[0]) < self.runtime:
        raise ValueError(
          f"input '{name}' has {int(tensor.shape[0])} timesteps < runtime {self.runtime}.")

  def _set_active(self, active: bool):
    # Fire on each play<->pause transition: lock the cameras (and resume the
    # follow window) while advancing, hand them back for zoom/pan while idle.
    if active == self._was_active:
      return
    for widget in self.widgets:
      widget.set_paused(not active)
    self._was_active = active

  def step(self, event):
    # Measure the actual steps/second over a rolling ~0.5s window and report it to
    # the panel. Done first (before any early return) so idle/finished states settle
    # back to 0 rather than showing a stale rate.
    now = time.perf_counter()
    if self._sps_t0 is None:
      self._sps_t0 = now
    elapsed = now - self._sps_t0
    if elapsed >= 0.5:
      self.panel.set_steps_per_second(self._sps_count / elapsed)
      self._sps_count = 0
      self._sps_t0 = now

    # Check if runtime is over
    if self.current_time >= self.runtime:
      self.timer.stop()
      # Hand the (bounded) cameras to the user now that the follow window is done.
      for widget in self.widgets:
        widget.finish()
      self.running = False
      self.panel.on_finish()
      self.canvas.update()
      return

    # Advance only when playing or there are queued steps; otherwise idle so the
    # user can zoom/pan the paused plots.
    active = self.running or self.step_budget > 0
    self._set_active(active)
    if not active:
      return

    t = self.current_time

    # Simulate one timestep in network
    tstep_inputs = {layer_name: layer_inputs[t] for layer_name, layer_inputs in self.inputs.items()}
    self.network.step(tstep_inputs, t)

    # Cheap per-step data capture into GPU buffers -- ALWAYS every step, so the data
    # is complete regardless of how often we draw.
    for widget in self.widgets:
      widget.capture(t)

    self._sps_count += 1   # count actual advancing steps for the rate readout

    manual = not self.running              # advancing via Step / Run N, not Play
    if self.step_budget > 0:
      self.step_budget -= 1

    # Throttle the expensive part: widget.render() (camera/axes/uniforms) + the
    # full-canvas redraw + buffer swap. Force a draw on the final queued manual step
    # so a Step / Run N result is shown immediately rather than at the next throttled
    # draw.
    force_draw = manual and self.step_budget == 0
    if force_draw or self._should_draw():
      for widget in self.widgets:
        widget.render(t)
      self.canvas.update()

    self.current_time += 1
    self.panel.set_time(self.current_time, self.runtime)

  def _should_draw(self):
    if self.draw_fps is None:
      return True
    now = time.perf_counter()
    if self._last_draw is None or (now - self._last_draw) >= 1.0 / self.draw_fps:
      self._last_draw = now
      return True
    return False

  def run(self, inputs: dict[str, torch.Tensor] | None = None, runtime: int = None):
    # The stimulus comes from the network's make_input(runtime) when it implements one
    # (regenerated on every reload so its shape tracks the parameters); otherwise a fixed
    # `inputs` dict must be supplied here. `runtime` sizes the full-history GL buffers.
    if runtime is None:
      raise ValueError("Application.run() requires `runtime`.")
    self.runtime = runtime
    if self._has_make_input:
      self.inputs = self.network.make_input(runtime)
    elif inputs is not None:
      self.inputs = inputs
    else:
      raise ValueError(
        "Application.run() needs an `inputs` dict unless the network implements "
        "make_input().")
    # Prime widgets now that runtime is known (full-history buffers need it).
    for widget in self.widgets:
      widget.prime(self.network, runtime)
    # Start paused: the timer ticks, but the sim only advances once the user hits
    # Play / Step / Run N (the plot cameras are interactive while idle).
    self.timer = app.Timer(
      interval=self._interval_for(self.max_steps_per_second), connect=self.step,
      start=True)
    # A separate ~60 Hz timer pumps the control panel's event loop when it has one
    # (the Qt window); the GLFW loop ticks both timers. The plots are unaffected.
    if self.panel.needs_pump:
      self.pump_timer = app.Timer(interval=1/60, connect=lambda e: self.panel.pump(), start=True)
    app.run()
    self.panel.shutdown()
