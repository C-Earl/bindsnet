from bindsnet.rendering.app import Application
from bindsnet.rendering.widgets import VoltagePlot, RasterPlot, WeightPlot
from model import create_model
import torch

SIM_TIME = 1000
BATCH_SIZE = 1
DEVICE = "cuda"
DRAW_FPS = 30          # cap plot redraws; the sim runs as fast as it can between draws

IN_SIZE = 100
EXC_SIZE = 20_000
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
app = Application(network, 2800, 1800, header="BindsNET Network Activity",
                  step_rate=99999999999, draw_fps=DRAW_FPS,
                  parameters={
                    "Input size": IN_SIZE,
                    "Excitatory size": EXC_SIZE,
                    "Inhibitory size": INH_SIZE,
                    "I -> EXC connectivity": I_TO_EXC_CONNECTIVITY,
                    "I -> INH connectivity": I_TO_INH_CONNECTIVITY,
                    "INH -> EXC connectivity": INH_TO_EXC_CONNECTIVITY,
                    "EXC -> INH connectivity": EXC_TO_INH_CONNECTIVITY,
                  })
inputs = {"I": torch.rand(SIM_TIME, BATCH_SIZE, IN_SIZE, device=DEVICE) > 0.90}
app.add_widget(
  RasterPlot(layer_name="EXC_LIF", max_timesteps=500),
  row=0, col=0,
)
app.add_widget(
  VoltagePlot(layer_name="EXC_LIF", neuron_ids=[i for i in range(100)], max_timesteps=500),
  row=0, col=1,
)
app.add_widget(
  # Heatmap of the I -> EXC weight matrix (source.n=100 rows x target.n=20000 cols)
  WeightPlot(source="I", target="EXC_LIF", feature_name="I_to_EXC_weight"),
  row=1, col=0,
)
app.run(inputs=inputs, runtime=SIM_TIME)
