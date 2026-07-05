from vispy import app, scene
import time
import torch
from bindsnet.rendering.widgets import AbstractWidget
from bindsnet.rendering.controls import QtControlPanel
from bindsnet.rendering.chrome_cache import ChromeCache
from bindsnet.network.network import GUINetwork

#### Backend ####
# Plots render on GLFW: draws straight to the window, drives the sim from a tight
# poll-loop -> full speed. vispy's Qt backend is much slower. Controls: see controls.py
app.use_app('glfw')


class Application():
  def __init__(self, network: GUINetwork,
               width=1400, height=900, title="BindsNET GUI",
               header: str | None = None,
               max_steps_per_second: int | float | str | None = None,
               draw_fps: float | None = None,
               parameters: dict | None = None):
    self.width, self.height = width, height

    #### Network lifecycle ####
    # If `network` overrides build() (inheritable model), the Application drives it:
    # build() assembles, make_input() supplies stimulus, network.parameters feeds the
    # panel, "Apply & Reload" rebuilds in place. Else used as-is, driven by run()'s dict.
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
    # Static chrome (axes/labels/titles) baked to a texture + blit each frame instead of
    # vispy re-processing every AxisVisual/TextVisual on the CPU (~80% of the draw).
    self.chrome_cache = None
    self.inputs = None      # network inputs; set by run()
    self.runtime = None     # total sim runtime; set by run()
    self.current_time = 0   # current timestep

    # Cap on sim steps/sec; `inf`/"max" = as fast as possible (0s timer interval).
    # Defaults to the draw rate (or 60).
    if max_steps_per_second is None:
      max_steps_per_second = draw_fps if draw_fps is not None else 60
    self.max_steps_per_second = self._coerce_sps(max_steps_per_second)

    # Decouple the expensive redraw from the sim rate: capture runs every step, redraw
    # fires at most `draw_fps`/sec. No data lost. draw_fps=None draws every step.
    self.draw_fps = draw_fps
    self._last_draw = None

    #### Simulation run-state (driven by the active control panel) ####
    # Timer always ticks; a tick advances the sim only when active.
    #   running     : continuous play (Play/Pause)
    #   step_budget : discrete steps queued (Step / Run N), consumed one per tick
    # active == running or step_budget > 0; else idle (cameras handed to the user).
    self.running = False
    self.step_budget = 0
    self._was_active = None   # last active-state; fires play<->pause transitions once

    ### Rolling ~2x/sec steps/second measurement ###
    self._sps_count = 0       # steps since the window opened
    self._sps_t0 = None       # perf_counter at window start

    ### VisPy canvas (GLFW) + grid layout ###
    self.canvas = scene.SceneCanvas(
      title=title,
      keys='interactive',
      bgcolor='black',
      size=(self.width, self.height),
      show=True,
    )
    # Optional centered title above the plot grid; spacer row keeps it (or the top tick
    # labels) off the canvas edge.
    self.layout = self.canvas.central_widget.add_grid(margin=0)
    next_row = 0
    self.layout.add_widget(row=next_row, col=0).height_max = 24    # top padding
    next_row += 1
    if header is not None:
      self.title_label = scene.Label(header, color='white', font_size=20, bold=True)
      self.title_label.height_max = 48
      self.layout.add_widget(self.title_label, row=next_row, col=0)
      next_row += 1
    else:
      self.title_label = None

    self.grid = self.layout.add_grid(row=next_row, col=0, margin=10)

    self.network.migrate()    # network tensors -> shared OpenGL buffers

    # Control surface (separate Qt window); calls back into toggle_play/step_once/run_n.
    self.panel = QtControlPanel(self, parameters=self.parameters)

  #### Steps-per-second rate ####
  @staticmethod
  def _coerce_sps(value: int | float | str) -> float:
    # Number, or "inf"/"max"/"unlimited"/"" -> as fast as possible.
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
    # inf steps/sec -> 0s interval (fires as fast as vispy can).
    return 0.0 if sps == float("inf") else 1.0 / sps

  def set_max_steps_per_second(self, value: int | float | str):
    # Swap the running timer's interval in place.
    self.max_steps_per_second = self._coerce_sps(value)
    if hasattr(self, "timer"):
      self.timer.interval = self._interval_for(self.max_steps_per_second)

  def add_widget(self, widget: AbstractWidget, row: int, col: int):
    self.widgets.append(widget)
    self.grid.add_widget(widget.grid, row, col)
    # Priming deferred to run() (full-history buffers need runtime); widgets added
    # after run() prime now.
    if self.runtime is not None:
      widget.prime(self.network, self.runtime)

  #### Control callbacks (panel-agnostic) ####
  def toggle_play(self):
    self.running = not self.running
    self.panel.set_playing(self.running)

  def step_once(self):
    self.step_budget += 1   # consumed next tick, even while paused

  def run_n(self, n: int):
    if n and n > 0:
      self.step_budget += int(n)

  def reset(self):
    # Clear live state + GL history, rewind to t=0, restore views, re-arm run-state.
    # Restart the timer in case the run had finished.
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
    self._set_active(False)   # re-lock cameras to the running state
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

    # Coerce fields first; a bad value aborts before the model is touched.
    try:
      values = self.panel.get_parameter_values()
    except Exception as exc:
      self.panel.show_status(f"Invalid parameter: {exc}", error=True)
      return

    self.canvas.set_current()   # GL calls below target the plot context
    try:
      self.network.rebuild(**values)   # free GL, clear, set params, re-run build()
      self.network.migrate()           # alloc rebuilt net's shared GL buffers
    except Exception as exc:
      # build() failed. rebuild() clears-then-builds, so a later good reload recovers.
      self.panel.show_status(f"Build failed: {exc}", error=True)
      return

    # Re-bind plots (release old visuals, alloc fresh buffers) before regen'ing inputs,
    # so a rare input mismatch can't leave widgets pointing at freed buffers.
    for widget in self.widgets:
      widget.reload(self.network)
    self.parameters = dict(self.network.parameters)

    # Regenerate stimulus. make_input() always fits; a fixed dict is warned if it doesn't.
    input_warning = None
    if self._has_make_input:
      self.inputs = self.network.make_input(self.runtime)
    else:
      try:
        self._validate_inputs(self.inputs, self.network)
      except Exception as exc:
        input_warning = f"Reloaded, but inputs no longer fit: {exc}"

    # Re-arm at t=0 (rebuilt net + history buffers already fresh/zeroed).
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
    # Each input's last dim must equal its layer's n; a time-major tensor must cover
    # the runtime. Raises ValueError on mismatch.
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
    # On each play<->pause transition: lock cameras while advancing, hand back idle.
    if active == self._was_active:
      return
    for widget in self.widgets:
      widget.set_paused(not active)
    # Advancing: bake + blit static chrome. Idle: show live dynamic chrome so it relabels.
    if self.chrome_cache is not None:
      if active:
        self.chrome_cache.enable()
      else:
        self.chrome_cache.disable()
    self._was_active = active

  def step(self, event):
    ### Rolling ~0.5s steps/second measurement ###
    # First (before early returns) so idle/finished settle to 0, not a stale rate.
    now = time.perf_counter()
    if self._sps_t0 is None:
      self._sps_t0 = now
    elapsed = now - self._sps_t0
    if elapsed >= 0.5:
      self.panel.set_steps_per_second(self._sps_count / elapsed)
      self._sps_count = 0
      self._sps_t0 = now

    ### Stop once runtime is over ###
    if self.current_time >= self.runtime:
      self.timer.stop()
      for widget in self.widgets:
        widget.finish()   # hand bounded cameras to the user
      # Live dynamic chrome so the finished run relabels on zoom/pan (_set_active isn't
      # called on finish).
      if self.chrome_cache is not None:
        self.chrome_cache.disable()
      self.running = False
      self.panel.on_finish()
      self.canvas.update()
      return

    ### Advance (only when playing or steps queued) ###
    active = self.running or self.step_budget > 0
    self._set_active(active)
    if not active:
      return   # idle: let the user zoom/pan

    t = self.current_time
    tstep_inputs = {layer_name: layer_inputs[t] for layer_name, layer_inputs in self.inputs.items()}
    self.network.step(tstep_inputs, t)

    # Cheap per-step capture -- ALWAYS every step, so the data is complete.
    for widget in self.widgets:
      widget.capture(t)

    self._sps_count += 1   # count advancing steps for the readout

    manual = not self.running              # advancing via Step / Run N, not Play
    if self.step_budget > 0:
      self.step_budget -= 1

    ### Draw (throttled) ###
    # render() + full redraw + swap is the expensive part. Force a draw on the last
    # manual step so Step / Run N shows immediately.
    force_draw = manual and self.step_budget == 0
    if force_draw or self._should_draw():
      for widget in self.widgets:
        widget.render(t)
      # Re-bake static chrome only if stale (domain grew, resize); else a cheap check.
      if self.chrome_cache is not None:
        self.chrome_cache.refresh()
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
    # Stimulus from make_input(runtime) if implemented (tracks params across reloads),
    # else a fixed `inputs` dict. `runtime` sizes the full-history GL buffers.
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
    for widget in self.widgets:
      widget.prime(self.network, runtime)
    # Chrome cache (widgets' chrome now exists); disabled until the sim advances.
    self.chrome_cache = ChromeCache(self.canvas, self.widgets, header=self.title_label)
    # Start paused: timer ticks, sim advances only on Play / Step / Run N.
    self.timer = app.Timer(
      interval=self._interval_for(self.max_steps_per_second), connect=self.step,
      start=True)
    # ~60 Hz timer pumps the Qt panel's event loop; GLFW ticks both timers.
    if self.panel.needs_pump:
      self.pump_timer = app.Timer(interval=1/60, connect=lambda e: self.panel.pump(), start=True)
    app.run()
    self.panel.shutdown()
