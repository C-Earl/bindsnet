from vispy import scene
from abc import abstractmethod
import numpy as np

from abc import abstractmethod


class AbstractWidget:
  def __init__(self, width: float, height: float, x:float, y:float):
    self.width = width      # Widget width
    self.height = height    # Widget height
    self.x = x              # Bottom-left x coordinate
    self.y = y              # Bottom-right y coordinate
    self.view = scene.widgets.ViewBox()

  @abstractmethod
  def prime(self, network):
    pass

  @abstractmethod
  def render(self):
    pass

class RasterPlot(AbstractWidget):
  def __init__(self, width: float, height: float, x:float, y:float,
               layer_name: str,
               max_timesteps: int):
    super().__init__(width, height, x, y)
    self.layer_name = layer_name
    self.layer = None   # Initialized after added to Application object
    self.spike_markers = None
    self.history = []

    self.view.camera = 'panzoom'
    self.markers = scene.visuals.Markers(
      parent=self.view.scene
    )

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    self.spike_markers = np.zeros((self.layer_size, 2), dtype=np.float32) # TODO: Can we make this smaller?

  def render(self, t):

    spike_data = self.layer.s.cpu().numpy()

    spike_ids = np.where(spike_data > 0)[1]

    for sid in spike_ids:
        self.history.append([t, sid])

    if len(self.history) == 0:
        return

    points = np.array(self.history, dtype=np.float32)

    self.markers.set_data(
        points,
        face_color='white',
        size=4
    )

    self.view.camera.set_range(
        x=(max(0, t - 100), max(100, t)),
        y=(0, self.layer_size)
    )
