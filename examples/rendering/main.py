from bindsnet.rendering.app import Application
from bindsnet.rendering.widgets import RasterPlot, VoltagePlot, AdvancedRasterPlot
from model import create_model
import torch

SIM_TIME = 1000
BATCH_SIZE = 1
DEVICE = "cuda"

IN_SIZE = 100
EXC_SIZE = 10_000
INH_SIZE = 2_000
I_TO_EXC_CONNECTIVITY = 0.15
I_TO_INH_CONNECTIVITY = 0.05
INH_TO_EXC_CONNECTIVITY = 0.05
EXC_TO_INH_CONNECTIVITY = 0.05

network = create_model(
  DEVICE,
  IN_SIZE,
  EXC_SIZE,
  INH_SIZE,
  I_TO_EXC_CONNECTIVITY,
  I_TO_INH_CONNECTIVITY,
  INH_TO_EXC_CONNECTIVITY,
  EXC_TO_INH_CONNECTIVITY,
)
app = Application(network, 1400, 900, step_rate=99999)
inputs = {"I" : torch.rand(SIM_TIME, BATCH_SIZE, IN_SIZE, device=DEVICE) > 0.90}
app.add_widget(
  AdvancedRasterPlot(
    layer_name="EXC_LIF",
    max_timesteps=500,
  ),
  row=0, col=0
)
# app.add_widget(
#   VoltagePlot(
#     width=700,
#     height=450,
#     x=50+700,
#     y=50,
#     neuron_ids=[10, 20],
#     layer_name="EXC_LIF",
#     max_timesteps=500,
#   ),
#   row=0,
#   col=0
# )
app.run(inputs=inputs, runtime=SIM_TIME)
