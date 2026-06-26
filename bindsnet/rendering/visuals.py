from vispy.visuals import ImageVisual, Visual
from vispy.scene.visuals import create_visual_node
from vispy import gloo
from vispy.gloo.context import get_current_canvas
from cuda.bindings import driver
import OpenGL.GL as gl
import logging
import torch
import numpy as np


class _UnsetUSpikesFilter(logging.Filter):
  # RasterHistoryVisual deliberately never sets its `u_spikes` samplerBuffer
  # (gloo has no samplerBuffer support; it defaults to texture unit 0, which we
  # bind by hand). VisPy's one-time program validation logs that as an "unset
  # variable" -- drop just that message so it doesn't look like an error.
  def filter(self, record):
    return 'u_spikes' not in record.getMessage()


logging.getLogger('vispy').addFilter(_UnsetUSpikesFilter())


def extract_gl_id(gl_object):
  canvas = get_current_canvas()   # TODO: Maybe a cleaner way to get canvas reference?
  gl_object_id = gl_object.id
  return canvas.context.shared.parser._objects[gl_object_id].handle


# Full-history, true-zero-copy spike raster. The layer's spikes are written in
# place by the node straight into a CUDA-registered GL buffer holding the WHOLE
# run, (total_timesteps, batch, n_neurons) one byte per spike (time-major). This
# visual binds that buffer as a TEXTURE_BUFFER and the fragment shader resolves
# each on-screen pixel (time, neuron) -> texelFetch -> spike colour. No per-frame
# copy, no ring/roll/seam: x is absolute time, so zooming out shows all history
# back to t=0.
#
# #version 140: gives texelFetch + usamplerBuffer while still allowing
# gl_FragColor and the attribute/varying qualifiers vispy's transform Function
# emits (vispy rewrites these to in/out to match the version automatically).
_RASTER_HIST_VERT = """
#version 140
attribute vec2 a_pos;     // quad corner in DATA coords: x=time [0,T], y=neuron [0,n]
varying vec2 v_data;      // interpolated data coord -> fragment
void main() {
    v_data = a_pos;
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
}
"""

_RASTER_HIST_FRAG = """
#version 140
varying vec2 v_data;
uniform usamplerBuffer u_spikes;   // R8UI history buffer; left UNSET -> texture unit 0
uniform int u_n;        // neurons displayed (layer n)
uniform int u_T;        // total timesteps in the buffer
uniform int u_stride;   // elements per timestep row (= batch*n)
uniform vec4 u_on;      // spike colour
uniform vec4 u_off;     // background colour
void main() {
    int time   = int(floor(v_data.x));
    int neuron = int(floor(v_data.y));
    if (time < 0 || time >= u_T || neuron < 0 || neuron >= u_n) discard;
    int idx = time * u_stride + neuron;          // time-major; batch 0
    uint s = texelFetch(u_spikes, idx).r;        // 0 or 1
    gl_FragColor = (s != 0u) ? u_on : u_off;
}
"""


class RasterHistoryVisual(Visual):
  def __init__(self, n_neurons, total_timesteps, row_stride, gl_buffer_id,
               on_color=(1.0, 1.0, 1.0, 1.0), off_color=(0.0, 0.0, 0.0, 1.0)):
    self.n = int(n_neurons)
    self.T = int(total_timesteps)
    self.stride = int(row_stride)        # = batch * n; idx = time*stride + neuron
    self._gl_buffer_id = int(gl_buffer_id)   # raw GL buffer with the spike history
    self._tbo_tex = None                 # GL texture viewing that buffer (lazy)
    Visual.__init__(self, vcode=_RASTER_HIST_VERT, fcode=_RASTER_HIST_FRAG)

    # One quad spanning the whole data region; the fragment shader does the work.
    corners = np.array([[0, 0], [self.T, 0], [0, self.n], [self.T, self.n]],
                       dtype=np.float32)
    self._pos_vbo = gloo.VertexBuffer(corners)
    self.shared_program['a_pos'] = self._pos_vbo
    self.shared_program['u_n'] = self.n
    self.shared_program['u_T'] = self.T
    self.shared_program['u_stride'] = self.stride
    self.shared_program['u_on'] = on_color
    self.shared_program['u_off'] = off_color
    # NOTE: u_spikes is deliberately never set. gloo has no samplerBuffer support,
    # and an unset sampler defaults to texture unit 0, which we bind in _prepare_draw.
    self._draw_mode = 'triangle_strip'
    self.set_gl_state('translucent', depth_test=False)

  def _create_tbo(self):
    # A buffer texture is a 1-D view of the GL buffer as R8UI texels. glTexBuffer
    # only references the buffer (no copy); texelFetch always reads its live bytes.
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R8UI, self._gl_buffer_id)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
    self._tbo_tex = tex

  def _prepare_draw(self, view):
    if self._tbo_tex is None:
      self._create_tbo()
    # Bind our buffer-texture to unit 0 with an IMMEDIATE raw GL call. gloo's own
    # draw is deferred to the canvas GLIR flush, but this binding persists until
    # then -- no other visual binds the GL_TEXTURE_BUFFER target -- and u_spikes
    # samples unit 0 by default. This is the one raw-GL touch the deferred GLIR
    # pipeline forces (gloo can't carry a samplerBuffer for us).
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self._tbo_tex)

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis == 0:
      return (0, self.T)
    if axis == 1:
      return (0, self.n)
    return None

  # No __del__: the buffer texture is freed when the GL context is destroyed.
  # Calling glDeleteTextures from __del__ races interpreter shutdown (PyOpenGL
  # lazily imports an array handler too late -> a noisy stderr message).


RasterHistory = create_visual_node(RasterHistoryVisual)


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
    # Reused gather destination so the per-frame index_select doesn't allocate.
    self._gather_out = torch.empty(n_neurons, dtype=torch.float32, device=idx.device)
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
    # Build every head's index buffer up front. Each is static (a given head's
    # seam sits at a fixed slot), so doing all W now turns the former per-frame
    # _make_index + IndexBuffer cost -- a ~hundreds-of-us stutter for the first W
    # frames -- into a one-time startup cost; _set_seam is a pure cache hit after.
    # No extra steady-state memory: the lazy cache already grew to W buffers.
    self._prebuild_seams()
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

  def _prebuild_seams(self):
    # Populate _ibo_cache for every possible head so the simulation-time path is
    # always a cache hit (see __init__). Leaves head 0 bound.
    for head in range(self.width):
      self._set_seam(head)
    self._set_seam(0)

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

  def capture_voltages(self, t):
    # CHEAP per-step data capture: copy this timestep's voltages into the ring
    # buffer. Must run EVERY step (else the trace gets gaps), independent of how
    # often we actually draw. Does NOT call update() -- see refresh().
    if self._cuda_res is None and not self._register():
      return  # buffer not on the GPU yet; skip this frame

    wrapped_t = t % self.width

    # gather ONLY the selected neurons, on-device -- no host copy, and no per-frame
    # allocation (reuse self._gather_out). re-fetch each frame: LIFNodes rebinds
    # layer.v to a new tensor every step.
    v = torch.index_select(self.volt_getter().reshape(-1), 0, self._idx, out=self._gather_out)

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

  def refresh(self, t):
    # DRAW-time only: point the roll at the newest captured column, break the seam
    # there, and request a redraw. All captured columns are already in the ring, so
    # this can run less often than capture without losing data.
    if self._cuda_res is None:
      return  # nothing captured yet
    wrapped_t = t % self.width
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

  Uses CUDA<->GL *texture* interop: a feature value is a snapshot (no time axis), so
  each refresh copies the WHOLE matrix into the texture with a single ``cuMemcpy2D``
  (device->array, no host roundtrip).

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
    # to the GPU, so registration is done lazily.
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
