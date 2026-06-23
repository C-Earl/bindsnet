from vispy.visuals import ImageVisual
from vispy.scene.visuals import create_visual_node
from vispy.gloo.context import get_current_canvas
from cuda.bindings import driver
import torch
import numpy as np
from vispy.visuals.shaders import Function

_ROLL_LOOKUP = """
    uniform float u_roll;
    vec4 rolled_lookup(vec2 texcoord) {
        vec2 tc = texcoord;
        tc.x = fract(tc.x + u_roll);   // ring-buffer roll; nearest filtering => no seam
        return texture2D($texture, tc);
    }
"""

def extract_gl_id(gl_object):
  canvas = get_current_canvas()   # TODO: Maybe a cleaner way to get canvas reference?
  gl_object_id = gl_object.id
  return canvas.context.shared.parser._objects[gl_object_id].handle


class RasterTextureVisual(ImageVisual):
  def __init__(self, layer_size, width, spike_tensor):
    self.layer_size = layer_size
    self.width = width
    self.spike_tensor = spike_tensor
    self._roll = 0.0  # current horizontal offset, persisted across rebuilds
    self._roll_fn = None
    dummy = np.zeros((layer_size, width), dtype=np.uint8)
    self._cuda_tex_resource = None
    super().__init__(data=dummy, texture_format=np.uint8, clim=(0, 1), cmap='grays')
    self.freeze()

  def _build_interpolation(self):
    # ImageVisual installs its own get_data here on the first draw, so we
    # wrap it AND reapply self._roll (otherwise the first build clobbers it).
    super()._build_interpolation()
    fn = Function(_ROLL_LOOKUP)
    fn['texture'] = self._texture
    self._data_lookup_fn = fn
    self._roll_fn = fn
    self.shared_program.frag['get_data'] = fn
    self.shared_program['u_roll'] = self._roll

  def _register_texture(self):
    # The gloo texture's GL object only exists after the first draw has
    # flushed it to the GPU, so this is done lazily.
    try:
      gl_tex_id = extract_gl_id(self._texture)
    except (KeyError, AttributeError):
      return False
    if not gl_tex_id:
      return False

    GL_TEXTURE_2D = 0x0DE1
    err, resource = driver.cuGraphicsGLRegisterImage(
      gl_tex_id,
      GL_TEXTURE_2D,
      0,  # CU_GRAPHICS_REGISTER_FLAGS_NONE
    )
    if err != 0:
      raise RuntimeError(f"cuGraphicsGLRegisterImage failed: {err}")
    self._cuda_tex_resource = resource
    return True

  def migrate_spikes(self, t: int):
    if self._cuda_tex_resource is None and not self._register_texture():
      return  # texture not on the GPU yet; skip this frame

    wrapped_t = t % self.width

    # Spike vector for this timestep: layer_size contiguous uint8 bytes
    # spikes = self.spike_tensor.contiguous()
    src_ptr = self.spike_tensor.data_ptr()

    res = self._cuda_tex_resource

    ### Texture becomes CUDA-owned ###
    (err,) = driver.cuGraphicsMapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"map texture failed: {err}")

    err, array = driver.cuGraphicsSubResourceGetMappedArray(res, 0, 0)
    if err != 0:
      raise RuntimeError(f"get mapped array failed: {err}")

    ### Copy column ###
    cp = driver.CUDA_MEMCPY2D()
    cp.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cp.srcDevice = src_ptr
    cp.srcPitch = 1
    cp.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
    cp.dstArray = array
    cp.dstXInBytes = wrapped_t  # column offset (1 byte/texel)
    cp.dstY = 0
    cp.WidthInBytes = 1  # one timestep wide
    cp.Height = self.layer_size  # all neurons tall
    (err,) = driver.cuMemcpy2D(cp)  # synchronous; `spikes` stays alive
    if err != 0:
      raise RuntimeError(f"cuMemcpy2D failed: {err}")

    ### Hand the texture back to OpenGL so VisPy can draw ###
    (err,) = driver.cuGraphicsUnmapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"unmap texture failed: {err}")

    ### Update the shader's roll offset so the texture appears to scroll ###
    self._roll = ((wrapped_t + 1) % self.width) / self.width
    if self._roll_fn is not None:
      self.shared_program['u_roll'] = self._roll
    self.update()

  def __del__(self):
    if self._cuda_tex_resource is not None:
      driver.cuGraphicsUnregisterResource(self._cuda_tex_resource)


RasterTexture = create_visual_node(RasterTextureVisual)
