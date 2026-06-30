from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight, Mask
from bindsnet.network.network import GUINetwork
from bindsnet.learning.MCC_learning import MSTDP
import torch


class ExampleNetwork(GUINetwork):
  # language=rst
  """
  Example inheritable GUINetwork: an Input layer projecting to excitatory + inhibitory
  LIF populations with recurrent inhibition. The model parameters are constructor
  arguments stored via :meth:`set_parameters`, so the Application's control panel can
  edit them and rebuild the network live ("Apply & Reload") -- :meth:`build` reassembles
  it from the current parameters and :meth:`make_input` regenerates the stimulus so its
  width tracks ``in_size``.
  """

  def __init__(self, device="cuda", in_size=100, exc_size=20_000, inh_size=2000,
               i_to_exc_connectivity=0.15, i_to_inh_connectivity=0.05,
               inh_to_exc_connectivity=0.05, exc_to_inh_connectivity=0.05):
    super().__init__()
    self.device = device   # config (not a tunable parameter): the GL render device
    # Declare the GUI-tunable parameters: stored in self.parameters (rendered as editable
    # rows in the control panel) AND set as attributes for build()/make_input().
    self.set_parameters(
      in_size=in_size,
      exc_size=exc_size,
      inh_size=inh_size,
      i_to_exc_connectivity=i_to_exc_connectivity,
      i_to_inh_connectivity=i_to_inh_connectivity,
      inh_to_exc_connectivity=inh_to_exc_connectivity,
      exc_to_inh_connectivity=exc_to_inh_connectivity,
    )

  def build(self):
    device = self.device
    self.add_layer(layer=Input(self.in_size), name='I')
    self.add_layer(layer=LIFNodes(self.exc_size), name='EXC_LIF')
    self.add_layer(layer=LIFNodes(self.inh_size), name='INH_LIF')
    self.add_connection(
      connection=MulticompartmentConnection(
        source=self.layers['I'],
        target=self.layers['EXC_LIF'],
        device=device,
        pipeline=[
          Weight(
            name='I_to_EXC_weight',
            value=torch.rand(self.in_size, self.exc_size, device=device),
            learning_rule=MSTDP,
            range=(0, 1)
          ),
          Mask(
            name='I_to_EXC_mask',
            value=torch.rand(self.in_size, self.exc_size, device=device)
                  > (1 - self.i_to_exc_connectivity),
          )
        ]),
      source='I',
      target='EXC_LIF')
    self.add_connection(
      connection=MulticompartmentConnection(
        source=self.layers['I'],
        target=self.layers['INH_LIF'],
        device=device,
        pipeline=[
          Weight(
            name='I_to_INH_weight',
            value=torch.rand(self.in_size, self.inh_size, device=device),
          ),
          Mask(
            name='I_to_INH_mask',
            value=torch.rand(self.in_size, self.inh_size, device=device)
                  > (1 - self.i_to_inh_connectivity),
          )
        ]),
      source='I',
      target='INH_LIF')
    self.add_connection(
      connection=MulticompartmentConnection(
        source=self.layers['INH_LIF'],
        target=self.layers['EXC_LIF'],
        device=device,
        pipeline=[
          Weight(
            name='INH_to_EXC_weight',
            value=-torch.rand(self.inh_size, self.exc_size, device=device),
          ),
          Mask(
            name='INH_to_EXC_mask',
            value=torch.rand(self.inh_size, self.exc_size, device=device)
                  > (1 - self.inh_to_exc_connectivity),
          )
        ]),
      source='INH_LIF',
      target='EXC_LIF')
    self.add_connection(
      connection=MulticompartmentConnection(
        source=self.layers['EXC_LIF'],
        target=self.layers['INH_LIF'],
        device=device,
        pipeline=[
          Weight(
            name='EXC_to_INH_weight',
            value=torch.rand(self.exc_size, self.inh_size, device=device),
          ),
          Mask(
            name='EXC_to_INH_mask',
            value=torch.rand(self.exc_size, self.inh_size, device=device)
                  > (1 - self.exc_to_inh_connectivity),
          )
        ]),
      source='EXC_LIF',
      target='INH_LIF')
    self.to(device)

  def make_input(self, runtime):
    # Poisson-ish random spike train into the input layer; width tracks in_size so the
    # stimulus always matches the (possibly rebuilt) network.
    return {"I": torch.rand(runtime, self.batch_size, self.in_size,
                            device=self.device) > 0.90}
