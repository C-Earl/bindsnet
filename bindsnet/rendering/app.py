from vispy import app, scene
import time
import torch
from bindsnet.rendering.widgets import AbstractWidget
from bindsnet.network.network import GUINetwork


class Application():
  def __init__(self, network: GUINetwork, width=1400, height=900, title="BindsNET GUI",
               header: str | None = None,
               step_rate: int | str = 500, draw_fps: float | None = None):
    self.width, self.height = width, height
    self.network = network
    self.widgets = []
    self.inputs = None      # Set when run() is called; Inputs into network during runtime
    self.runtime = None     # Set when run() is called; Total runtime of network simulation
    self.current_time = 0   # Current timestep in network; incremented during runtime
    self.step_rate = 1/step_rate

    # Decouple the (expensive) full-canvas redraw from the simulation rate: run the
    # sim + cheap per-step data capture (widget.capture) every step, but redraw
    # (widget.render + canvas redraw + swap) at most `draw_fps` times/second. No
    # data is lost because capture is independent of drawing. draw_fps=None draws
    # every step.
    self.draw_fps = draw_fps
    self._last_draw = None

    # Initialize VisPy canvas and grid layout for widget rendering
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
    if header is not None:
      self.title_label = scene.Label(header, color='white', font_size=20, bold=True)
      self.title_label.height_max = 48
      self.layout.add_widget(self.title_label, row=next_row, col=0)
    else:
      self.title_label = None
      self.layout.add_widget(row=next_row, col=0).height_max = 12   # top padding
    next_row += 1

    self.grid = self.layout.add_grid(row=next_row, col=0, margin=10)

    # Migrate network tensors to shared OpenGL buffers
    network.migrate()

  def add_widget(self, widget: AbstractWidget, row: int, col: int):
    self.widgets.append(widget)
    self.grid.add_widget(widget.grid, row, col)
    # Priming is deferred to run(): some widgets (full-history raster) need the
    # total runtime to size their GPU buffers, and runtime isn't known until run().
    # Support adding widgets after run() too, in which case prime immediately.
    if self.runtime is not None:
      widget.prime(self.network, self.runtime)

  def step(self, event):
    # Check if runtime is over
    if self.current_time >= self.runtime:
      self.timer.stop()
      # Hand the (bounded) cameras to the user now that the follow window is done.
      for widget in self.widgets:
        widget.finish()
      self.canvas.update()
      return

    # Simulate one timestep in network
    tstep_inputs = {layer_name: layer_inputs[self.current_time] for layer_name, layer_inputs in self.inputs.items()}
    self.network.step(tstep_inputs, self.current_time)

    # Cheap per-step data capture into GPU buffers -- ALWAYS every step, so the data
    # is complete regardless of how often we draw.
    for widget in self.widgets:
      widget.capture(self.current_time)

    # Throttle the expensive part: widget.render() (camera/axes/uniforms) + the
    # full-canvas redraw + buffer swap. render() schedules its own redraw, so it is
    # gated together with canvas.update().
    if self._should_draw():
      for widget in self.widgets:
        widget.render(self.current_time)
      self.canvas.update()

    self.current_time += 1

  def _should_draw(self):
    if self.draw_fps is None:
      return True
    now = time.perf_counter()
    if self._last_draw is None or (now - self._last_draw) >= 1.0 / self.draw_fps:
      self._last_draw = now
      return True
    return False

  def run(self, inputs: dict[str, torch.Tensor], runtime: int):
    self.inputs = inputs
    self.runtime = runtime
    # Prime widgets now that runtime is known (full-history buffers need it).
    for widget in self.widgets:
      widget.prime(self.network, runtime)
    self.timer = app.Timer(interval=self.step_rate, connect=self.step, start=True)
    app.run()
