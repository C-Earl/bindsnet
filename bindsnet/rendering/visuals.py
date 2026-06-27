from vispy.visuals import ImageVisual, Visual
from vispy.scene.visuals import create_visual_node
from vispy import gloo
from vispy.gloo.context import get_current_canvas
from cuda.bindings import driver
import OpenGL.GL as gl
import logging
import numpy as np


class _UnsetSamplerBufferFilter(logging.Filter):
  # RasterHistoryVisual / VoltageHistoryVisual deliberately never set their
  # samplerBuffer uniforms (`u_spikes`, `u_volts`): gloo has no samplerBuffer
  # support, so each defaults to texture unit 0, which we bind by hand. VisPy's
  # one-time program validation logs that as an "unset variable" -- drop just
  # those messages so they don't look like errors.
  def filter(self, record):
    msg = record.getMessage()
    return 'u_spikes' not in msg and 'u_volts' not in msg


logging.getLogger('vispy').addFilter(_UnsetSamplerBufferFilter())


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

// Cap on texels sampled per axis per pixel. At moderate zoom a pixel covers a
// handful of cells and we read them all; at extreme zoom-out we'd cover more
// than this, so we step (subsample) to bound cost -- some spikes may still be
// dropped only when one pixel spans >MAXSTEPS cells, far past the buggy regime.
const int MAXSTEPS = 64;

void main() {
    // Data-space size of one screen pixel (axis-aligned ortho -> fwidth is the
    // per-pixel extent along each data axis). This is the footprint we must
    // max-reduce over so sparse spikes survive when many cells map to one pixel.
    vec2 px = fwidth(v_data);

    int t0 = int(floor(v_data.x - 0.5 * px.x));
    int t1 = int(floor(v_data.x + 0.5 * px.x));
    int n0 = int(floor(v_data.y - 0.5 * px.y));
    int n1 = int(floor(v_data.y + 0.5 * px.y));

    t0 = clamp(t0, 0, u_T - 1);
    t1 = clamp(t1, 0, u_T - 1);
    n0 = clamp(n0, 0, u_n - 1);
    n1 = clamp(n1, 0, u_n - 1);

    // Discard pixels whose centre is fully outside the data region.
    if (v_data.x < 0.0 || v_data.x >= float(u_T) ||
        v_data.y < 0.0 || v_data.y >= float(u_n)) discard;

    int tstep = max(1, (t1 - t0 + 1 + MAXSTEPS - 1) / MAXSTEPS);
    int nstep = max(1, (n1 - n0 + 1 + MAXSTEPS - 1) / MAXSTEPS);

    // OR-reduce: any spike in this pixel's footprint lights it up.
    uint hit = 0u;
    for (int t = t0; t <= t1; t += tstep) {
        int row = t * u_stride;                       // time-major; batch 0
        for (int neuron = n0; neuron <= n1; neuron += nstep) {
            hit |= texelFetch(u_spikes, row + neuron).r;
        }
    }
    gl_FragColor = (hit != 0u) ? u_on : u_off;
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


# Full-history, zero-copy voltage traces -- the voltage analogue of
# RasterHistoryVisual. The layer's voltage is written in place by the node straight
# into a CUDA-registered GL buffer holding the WHOLE run, (T, batch, n) float32
# time-major (see GUINetwork.enable_voltage_history). This visual binds that buffer
# as a TEXTURE_BUFFER (R32F) and the VERTEX shader pulls v[t, neuron] via texelFetch
# to position each point of each selected neuron's trace. No per-frame copy, no ring
# buffer/roll/seam: x is absolute time, so zooming out shows all history back to t=0.
#
# #version 140: gives texelFetch + samplerBuffer in the vertex stage while still
# allowing gl_FragColor and the attribute/varying qualifiers vispy emits.
_VOLT_HIST_VERT = """
#version 140
attribute float a_time;     // absolute timestep for this vertex, 0..T-1
attribute float a_neuron;   // neuron id within the layer (which trace)
attribute vec4  a_color;    // per-neuron trace colour, static
uniform samplerBuffer u_volts;   // R32F history buffer; left UNSET -> texture unit 0
uniform int u_stride;            // floats per timestep row (= batch*n)
varying vec4 v_color;
void main() {
    int idx = int(a_time) * u_stride + int(a_neuron);   // time-major; batch 0
    float v = texelFetch(u_volts, idx).r;
    gl_Position = $transform(vec4(a_time, v, 0.0, 1.0));
    v_color = a_color;
}
"""

_VOLT_HIST_FRAG = """
#version 140
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class VoltageHistoryVisual(Visual):
  def __init__(self, neuron_ids, total_timesteps, row_stride, gl_buffer_id, colors):
    self.ids = [int(i) for i in neuron_ids]
    self.K = len(self.ids)
    self.T = int(total_timesteps)
    self.stride = int(row_stride)            # = batch * n; idx = time*stride + neuron
    self._gl_buffer_id = int(gl_buffer_id)   # raw GL buffer with the voltage history
    self._tbo_tex = None                     # GL texture viewing that buffer (lazy)
    Visual.__init__(self, vcode=_VOLT_HIST_VERT, fcode=_VOLT_HIST_FRAG)

    # One vertex per (selected neuron, timestep). The vertex shader pulls the y value
    # (voltage) from the buffer texture; only the static (time, neuron, colour) live
    # in attributes. K * T vertices total.
    times = np.tile(np.arange(self.T, dtype=np.float32), self.K)
    neurons = np.repeat(np.array(self.ids, dtype=np.float32), self.T)
    cols = np.repeat(colors.astype(np.float32), self.T, axis=0)
    self._time_vbo = gloo.VertexBuffer(times)
    self._neuron_vbo = gloo.VertexBuffer(neurons)
    self._color_vbo = gloo.VertexBuffer(cols)
    self.shared_program['a_time'] = self._time_vbo
    self.shared_program['a_neuron'] = self._neuron_vbo
    self.shared_program['a_color'] = self._color_vbo
    self.shared_program['u_stride'] = self.stride
    # NOTE: u_volts is deliberately never set. gloo has no samplerBuffer support, and
    # an unset sampler defaults to texture unit 0, which we bind in _prepare_draw.

    # Static GL_LINES segments joining consecutive timesteps within each neuron's
    # trace -- the seam-free analogue of ScrollLine's ring index (x is absolute time
    # now, so there is no wrap/seam). Built once.
    self._index_buffer = gloo.IndexBuffer(self._make_index())
    self._draw_mode = 'lines'
    self.set_gl_state('translucent', depth_test=False)

  def _make_index(self):
    # Edges (t, t+1) within each neuron's contiguous block of T vertices.
    T, K = self.T, self.K
    i = np.arange(T - 1)
    seg = np.stack([i, i + 1], axis=1)                       # (T-1, 2)
    return (seg[None] + (np.arange(K) * T)[:, None, None]).reshape(-1, 2).astype(np.uint32)

  def _create_tbo(self):
    # A buffer texture is a 1-D view of the GL buffer as R32F texels. glTexBuffer
    # only references the buffer (no copy); texelFetch always reads its live floats.
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R32F, self._gl_buffer_id)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
    self._tbo_tex = tex

  def _prepare_draw(self, view):
    if self._tbo_tex is None:
      self._create_tbo()
    # Bind our buffer-texture to unit 0 with an IMMEDIATE raw GL call. gloo flushes
    # each program's draw at the end of Program.draw(), so this binding persists
    # through THIS visual's own draw -- u_volts samples unit 0 by default. (The
    # raster does the same in its own _prepare_draw; per-draw flush keeps them from
    # colliding even though both target GL_TEXTURE_BUFFER on unit 0.)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self._tbo_tex)

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis == 0:
      return (0, self.T)
    return None   # y (voltage) bounds are data-dependent; the widget sets the camera

  # No __del__: the buffer texture is freed when the GL context is destroyed (see
  # the note on RasterHistoryVisual).


VoltageHistory = create_visual_node(VoltageHistoryVisual)


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
