from vispy.visuals import ImageVisual, Visual
from vispy import gloo
from vispy.scene.visuals import create_visual_node
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
from vispy.visuals._scalable_textures import GPUScaledTexture2D
from vispy import gloo
from vispy.gloo.context import get_current_canvas
from vispy.visuals import Visual
import numpy as np


def extract_gl_id(gl_object):
  canvas = get_current_canvas()   # TODO: Maybe a cleaner way to get canvas reference?
  gl_object_id = gl_object.id
  return canvas.context.shared.parser._objects[gl_object_id].handle


class RasterTextureVisual(ImageVisual):

  def __init__(self, layer_size, width, spike_vbo):
    self.layer_size = layer_size
    self.width = width
    self.spike_vbo = spike_vbo
    dummy_data = np.zeros((layer_size, width), dtype=np.uint8)
    super().__init__(data=dummy_data, texture_format=np.uint8, clim=(0, 1))

  def set_data(self, data):
    super().set_data(data)

  def migrate_spikes(self, t: int):
    if t == 0:
      return
    # self.set_data(((np.random.rand(self.layer_size, self.width) < 0.5) * 255).astype(np.uint8))
    wrapped_t = t % self.width
    gl.glBindBuffer(  # Bind BindsNET/PyTorch spike buffer
      gl.GL_PIXEL_UNPACK_BUFFER,
      self.spike_vbo
    )
    gl.glPixelStorei(  # Set alignment for unpacking to 1 byte ie, uint8
      gl.GL_UNPACK_ALIGNMENT,
      1
    )
    texture_gl_id = extract_gl_id(self._texture)
    gl.glBindTexture(  # Bind spike raster texture
      gl.GL_TEXTURE_2D,
      texture_gl_id
    )
    gl.glBindTexture(  # Bind spike raster texture
      gl.GL_TEXTURE_2D,
      texture_gl_id
    )
    gl.glTexSubImage2D(  # Copy spike data from buffer to texture
      gl.GL_TEXTURE_2D,
      0,
      wrapped_t,
      0,
      1,
      self.layer_size,
      gl.GL_RED,
      gl.GL_UNSIGNED_BYTE,
      None
    )
    self.update()

  # def _prepare_draw(self, view):
  #   super()._prepare_draw(view)
  #
  # def draw(self):
  #   super().draw()

  # def draw(self, event):
  #   # self.program.draw("triangles")
  #   super().draw(event)

RasterTexture = create_visual_node(RasterTextureVisual)