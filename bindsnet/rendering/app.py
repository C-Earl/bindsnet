from vispy import app, scene
import torch
from bindsnet.rendering.widgets import AbstractWidget
from bindsnet.network.network import GUINetwork


class Application():
  def __init__(self, network: GUINetwork, width=1400, height=900, title="BindsNET GUI",
               step_rate: int | str = 500):
    self.width, self.height = width, height
    self.network = network
    self.widgets = []
    self.inputs = None      # Set when run() is called; Inputs into network during runtime
    self.runtime = None     # Set when run() is called; Total runtime of network simulation
    self.current_time = 0   # Current timestep in network; incremented during runtime
    self.step_rate = step_rate if type(step_rate) == str \
      else 1/step_rate # Rate in hz to step network and update renders

    # Initialize VisPy canvas and grid layout for widget rendering
    self.canvas = scene.SceneCanvas(
      title=title,
      keys='interactive',
      bgcolor='black',
      size=(self.width, self.height),
      show=True
    )
    self.grid = self.canvas.central_widget.add_grid()

  def add_widget(self, widget: AbstractWidget, row: int, col: int):
    self.widgets.append(widget)
    widget.prime(self.network)    # Needed to initialize network-dependent widget variables
    self.grid.add_widget(widget.grid, row, col)

  def step(self, event):
    # Check if runtime is over
    if self.current_time >= self.runtime:
      self.timer.stop()
      return

    # Simulate one timestep in network
    tstep_inputs = {layer_name: layer_inputs[self.current_time] for layer_name, layer_inputs in self.inputs.items()}
    self.network.step(tstep_inputs)

    # Update widget renders
    for widget in self.widgets:
      widget.render(self.current_time)

    # Increment time
    self.current_time += 1

  def run(self, inputs: dict[str, torch.Tensor], runtime: int):
    self.inputs = inputs
    self.runtime = runtime
    self.timer = app.Timer(interval=self.step_rate, connect=self.step, start=True)
    app.run()
