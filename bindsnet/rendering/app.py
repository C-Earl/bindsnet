from vispy import app, scene
import torch
from bindsnet.rendering.widgets import AbstractWidget
from bindsnet.network.network import GUINetwork


class Application():
  def __init__(self, network: GUINetwork, width=1400, height=900, title="BindsNET GUI"):
    self.width, self.height = width, height
    self.network = network
    self.widgets = []
    self.inputs = None    # Set when run() is called
    self.current_time = 0

    self.canvas = scene.SceneCanvas(
      keys='interactive',
      bgcolor='black',
      size=(self.width, self.height),
      show=True
    )
    self.grid = self.canvas.central_widget.add_grid()

  def add_widget(self, widget: AbstractWidget, row: int, col: int):
    self.widgets.append(widget)
    widget.prime(self.network)
    self.grid.add_widget(widget.view, row, col)

  def prime_widgets(self):
    for widget in self.widgets:
      widget.prime()

  def step(self, event):
    # Simulate one timestep in network
    tstep_inputs = {layer_name: layer_inputs[self.current_time] for layer_name, layer_inputs in self.inputs.items()}
    self.network.step(tstep_inputs)

    # Update widget renders
    for widget in self.widgets:
      widget.render(self.current_time)

    # Increment time
    self.current_time += 1

  def run(self, inputs: dict[str, torch.Tensor], time):
    self.inputs = inputs
    self.timer = app.Timer(interval=0.00016, connect=self.step, start=True)
    app.run()
