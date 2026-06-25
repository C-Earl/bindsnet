from vispy.visuals import ImageVisual, Visual
from vispy.scene.visuals import create_visual_node
from vispy import gloo
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


_SCROLL_VERT = """
attribute float a_slot;     // 0..W-1, static
attribute float a_volt;     // voltage, CUDA-written ring buffer
attribute vec4  a_color;    // per-neuron color, static
uniform float u_head;       // current write head (wrapped_t)
uniform float u_width;      // W
varying vec4 v_color;
void main() {
    float xd = mod(a_slot - (u_head + 1.0), u_width);   // newest -> right edge, like the raster roll
    gl_Position = $transform(vec4(xd, a_volt, 0.0, 1.0));
    v_color = a_color;
}
"""

_SCROLL_FRAG = """
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class ScrollLineVisual(Visual):
  def __init__(self, n_neurons, width, volt_getter, idx, colors):
    self.n_neurons = n_neurons
    self.width = width
    self.volt_getter = volt_getter      # callable -> live layer.v (rebound each step)
    self._idx = idx                     # torch LongTensor of selected neuron ids (on device)
    self._cuda_res = None
    Visual.__init__(self, vcode=_SCROLL_VERT, fcode=_SCROLL_FRAG)

    N = n_neurons * width
    self._slot_vbo = gloo.VertexBuffer(np.tile(np.arange(width, dtype=np.float32), n_neurons))
    self._volt_vbo = gloo.VertexBuffer(np.zeros(N, dtype=np.float32))   # the CUDA-written ring
    self._color_vbo = gloo.VertexBuffer(np.repeat(colors.astype(np.float32), width, axis=0))
    self.shared_program['a_slot'] = self._slot_vbo
    self.shared_program['a_volt'] = self._volt_vbo
    self.shared_program['a_color'] = self._color_vbo
    self.shared_program['u_head'] = 0.0
    self.shared_program['u_width'] = float(width)

    self._ibo_cache = {}   # head -> IndexBuffer; each head's seam is static, built once
    self._set_seam(0)
    self._draw_mode = 'lines'
    self.set_gl_state('translucent', depth_test=False)

  def _make_index(self, head):
    # ring edges (i, i+1) for every slot EXCEPT the seam at the write head,
    # tiled per neuron. GL_LINES with shared endpoints draws a continuous trace.
    W, K = self.width, self.n_neurons
    i = np.arange(W); i = i[i != head]
    seg = np.stack([i, (i + 1) % W], axis=1)
    return (seg[None] + (np.arange(K) * W)[:, None, None]).reshape(-1, 2).astype(np.uint32)

  def _set_seam(self, head):
    # The index buffer for a given head never changes (the seam sits at a fixed
    # slot), and head = t % width only takes W distinct values -- so build each
    # head's buffer once and just rebind it thereafter. No per-frame NumPy
    # rebuild or GPU re-upload (the old hot spot). Caches W buffers:
    # W * K * (W-1) * 2 * 4 bytes; e.g. 100x100 -> ~7.7 MB.
    ibo = self._ibo_cache.get(head)
    if ibo is None:
      ibo = gloo.IndexBuffer(self._make_index(head))
      self._ibo_cache[head] = ibo
    self._index_buffer = ibo

  def _register(self):
    # The gloo VBO's GL object only exists after the first draw, so register lazily.
    try:
      gl_id = extract_gl_id(self._volt_vbo)
    except (KeyError, AttributeError):
      return False
    if not gl_id:
      return False
    # Flags=0 (NONE), NOT WRITE_DISCARD: we keep W-1 columns and write only one.
    err, res = driver.cuGraphicsGLRegisterBuffer(buffer=gl_id, Flags=0)
    if err != 0:
      raise RuntimeError(f"cuGraphicsGLRegisterBuffer failed: {err}")
    self._cuda_res = res
    return True

  def migrate_voltages(self, t):
    if self._cuda_res is None and not self._register():
      return  # buffer not on the GPU yet; skip this frame

    wrapped_t = t % self.width

    # gather ONLY the selected neurons, on-device -- no host copy.
    # re-fetch each frame: LIFNodes rebinds layer.v to a new tensor every step.
    v = self.volt_getter().reshape(-1).index_select(0, self._idx)

    res = self._cuda_res

    ### Buffer becomes CUDA-owned ###
    (err,) = driver.cuGraphicsMapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"map buffer failed: {err}")

    err, ptr, _ = driver.cuGraphicsResourceGetMappedPointer(res)
    if err != 0:
      raise RuntimeError(f"get mapped pointer failed: {err}")

    ### Scatter K voltages into column `wrapped_t` (neuron-major: stride W floats) ###
    cp = driver.CUDA_MEMCPY2D()
    cp.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cp.srcDevice = v.data_ptr()
    cp.srcPitch = 4
    cp.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cp.dstDevice = int(ptr) + 4 * wrapped_t
    cp.dstPitch = 4 * self.width
    cp.WidthInBytes = 4              # one float (y) per neuron
    cp.Height = self.n_neurons      # one row per selected neuron
    (err,) = driver.cuMemcpy2D(cp)  # synchronous; `v` stays alive
    if err != 0:
      raise RuntimeError(f"cuMemcpy2D failed: {err}")

    ### Hand the buffer back to OpenGL so VisPy can draw ###
    (err,) = driver.cuGraphicsUnmapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"unmap buffer failed: {err}")

    ### Roll so the newest column sits at the right edge, and break the seam ###
    self.shared_program['u_head'] = float(wrapped_t)
    self._set_seam(wrapped_t)
    self.update()

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _prepare_draw(self, view):
    pass

  def _compute_bounds(self, axis, view):
    return (0, self.width) if axis == 0 else None

  def __del__(self):
    if self._cuda_res is not None:
      driver.cuGraphicsUnregisterResource(self._cuda_res)


ScrollLine = create_visual_node(ScrollLineVisual)


class FeatureMatrixVisual(ImageVisual):
  # language=rst
  """
  Renders a connection feature's ``value`` matrix (shape ``(source_n, target_n)``)
  as a live heatmap, kept entirely on the GPU.

  Mirrors :class:`RasterTextureVisual`'s CUDA<->GL *texture* interop, but a feature
  value is a snapshot (no time axis) rather than a rolling time series, so there is
  no ring-buffer roll/seam handling: each refresh copies the WHOLE matrix into the
  texture with a single ``cuMemcpy2D`` (device->array, no host roundtrip).

  The dtype/clim/cmap are parameters so the same visual serves any feature
  (weights, mask, probability, ...); the owning widget picks them.
  """

  def __init__(self, rows, cols, value_getter,
               texture_format=np.float32, clim=(-1.0, 1.0), cmap='coolwarm'):
    self.rows = rows                  # = source.n  -> texture height / y axis
    self.cols = cols                  # = target.n  -> texture width  / x axis
    self.value_getter = value_getter  # callable -> live feature.value (device tensor)
    self._cuda_tex_resource = None
    dummy = np.zeros((rows, cols), dtype=texture_format)
    # Explicit numeric clim (NOT 'auto'): GPU-scaled textures freeze 'auto' clim on
    # the first (all-zero) upload, mapping everything to one color. See memory note.
    super().__init__(data=dummy, texture_format=texture_format, clim=clim, cmap=cmap)
    self.freeze()

  def _register_texture(self):
    # The gloo texture's GL object only exists after the first draw has flushed it
    # to the GPU, so registration is done lazily (same as RasterTextureVisual).
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

  def migrate(self):
    # Push the feature's current value matrix into the texture. No `t`: a feature
    # value is a live snapshot, not a function of the timestep.
    if self._cuda_tex_resource is None and not self._register_texture():
      return  # texture not on the GPU yet; skip this frame

    # Re-fetch each frame: learning rules may rebind feature.value to a new tensor
    # (e.g. Weight.compute under enforce_polarity), like LIFNodes rebinds layer.v.
    val = self.value_getter().contiguous()
    elem = val.element_size()
    src_ptr = val.data_ptr()
    res = self._cuda_tex_resource

    ### Texture becomes CUDA-owned ###
    (err,) = driver.cuGraphicsMapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"map texture failed: {err}")

    err, array = driver.cuGraphicsSubResourceGetMappedArray(res, 0, 0)
    if err != 0:
      raise RuntimeError(f"get mapped array failed: {err}")

    ### Copy the full matrix (row-major: cols fastest, so one row == one texture row) ###
    cp = driver.CUDA_MEMCPY2D()
    cp.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
    cp.srcDevice = src_ptr
    cp.srcPitch = self.cols * elem
    cp.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
    cp.dstArray = array
    cp.dstXInBytes = 0
    cp.dstY = 0
    cp.WidthInBytes = self.cols * elem  # full row
    cp.Height = self.rows               # all source neurons
    (err,) = driver.cuMemcpy2D(cp)      # synchronous; `val` stays alive
    if err != 0:
      raise RuntimeError(f"cuMemcpy2D failed: {err}")

    ### Hand the texture back to OpenGL so VisPy can draw ###
    (err,) = driver.cuGraphicsUnmapResources(1, res, 0)
    if err != 0:
      raise RuntimeError(f"unmap texture failed: {err}")

    self.update()

  def __del__(self):
    if self._cuda_tex_resource is not None:
      driver.cuGraphicsUnregisterResource(self._cuda_tex_resource)


FeatureMatrix = create_visual_node(FeatureMatrixVisual)
