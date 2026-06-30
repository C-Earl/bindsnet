from bindsnet.rendering.app import Application
from bindsnet.rendering.widgets import VoltagePlot, RasterPlot, WeightPlot, NetworkPlot
from model import ExampleNetwork

SIM_TIME = 1000
DEVICE = "cuda"
DRAW_FPS = 30          # cap plot redraws; the sim runs as fast as it can between draws

# An inheritable GUINetwork: its constructor stores the model parameters, build()
# assembles the network and make_input() generates the stimulus. The Application drives
# all three -- it builds the network, populates the control panel from net.parameters,
# and the "Apply & Reload" button re-runs build()/make_input() with the edited values.
network = ExampleNetwork(device=DEVICE)
app = Application(network, 2800, 1800, header="BindsNET Network Activity",
                  max_steps_per_second=float("inf"), draw_fps=DRAW_FPS)

app.add_widget(
  RasterPlot(layer_name="EXC_LIF", window_size=500),
  row=0, col=0,
)
app.add_widget(
  VoltagePlot(layer_name="EXC_LIF", neuron_ids=[i for i in range(100)], window_size=500),
  row=0, col=1,
)
app.add_widget(
  RasterPlot(layer_name="INH_LIF", window_size=500),
  row=1, col=0,
)
app.add_widget(
  VoltagePlot(layer_name="INH_LIF", neuron_ids=[i for i in range(100)], window_size=500),
  row=1, col=1,
)
app.add_widget(
  # Heatmap of the I -> EXC weight matrix (source.n=100 rows x target.n=20000 cols)
  WeightPlot(source="I", target="EXC_LIF", feature_name="I_to_EXC_weight"),
  row=2, col=0,
)
# app.add_widget(
#   # The network itself: neurons as circles in layered columns (I / EXC / INH),
#   # synapses as weight-coloured lines (capped per connection), firing shown live.
#   NetworkPlot(afterglow=10),
#   row=1, col=1,
# )
app.run(runtime=SIM_TIME)
