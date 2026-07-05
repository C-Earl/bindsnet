from vispy.visuals import ImageVisual, Visual
from vispy.visuals.text.text import (TextVisual, _VERTEX_SHADER as _TEXT_VERT,
                                     _FRAGMENT_SHADER as _TEXT_FRAG)
from vispy.scene.visuals import create_visual_node
from vispy.visuals.transforms import NullTransform
from vispy import gloo
from vispy.gloo.context import get_current_canvas
try:
    from cuda.bindings import driver   # CUDA<->GL interop; absent on CPU-only installs
except Exception:
    driver = None
import OpenGL.GL as gl
import logging
import numpy as np


class _UnsetSamplerBufferFilter(logging.Filter):
  # Raster/Voltage/NeuronCloud never set their samplerBuffer uniforms (gloo has no
  # samplerBuffer -> default to unit 0, bound by hand). Drop vispy's "unset variable" log.
  _UNSET = ('u_spikes', 'u_volts', 'u_fire')

  def filter(self, record):
    msg = record.getMessage()
    return not any(name in msg for name in self._UNSET)


logging.getLogger('vispy').addFilter(_UnsetSamplerBufferFilter())


def extract_gl_id(gl_object):
  canvas = get_current_canvas()   # TODO: Maybe a cleaner way to get canvas reference?
  gl_object_id = gl_object.id
  return canvas.context.shared.parser._objects[gl_object_id].handle


#### Full-history, true-zero-copy spike raster ####
# Node writes spikes in place into a CUDA-registered GL buffer for the WHOLE run,
# (T, batch, n) one byte/spike (time-major). Bound as a TEXTURE_BUFFER; frag shader maps
# pixel (time, neuron) -> texelFetch -> colour. No per-frame copy, x = absolute time.
# #version 140: texelFetch + usamplerBuffer, still allows gl_FragColor + vispy's qualifiers.
_RASTER_HIST_VERT = """
#version 140
attribute vec2 a_pos;     // quad corner in DATA coords: x=time [0,T], y=neuron [0,n]
uniform float u_xoff;     // scroll offset (timesteps): shifts the quad LEFT on screen
                          // under a fixed camera, so the data scrolls without moving
                          // the camera (no transform cascade). v_data stays ABSOLUTE
                          // so the fragment shader still texelFetches the right column.
varying vec2 v_data;      // interpolated data coord -> fragment
void main() {
    v_data = a_pos;
    gl_Position = $transform(vec4(a_pos.x - u_xoff, a_pos.y, 0.0, 1.0));
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
    self.shared_program['u_xoff'] = 0.0      # no scroll until the widget drives it
    # u_spikes never set: unset sampler -> unit 0, bound in _prepare_draw.
    self._draw_mode = 'triangle_strip'
    self.set_gl_state('translucent', depth_test=False)

  def set_x_offset(self, x0):
    # Scroll under a fixed camera: one scalar-uniform write, no camera move.
    self.shared_program['u_xoff'] = float(x0)

  def _create_tbo(self):
    # 1-D R8UI view of the GL buffer; glTexBuffer references it (no copy).
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R8UI, self._gl_buffer_id)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
    self._tbo_tex = tex

  def _prepare_draw(self, view):
    if self._tbo_tex is None:
      self._create_tbo()
    # Immediate raw-GL bind to unit 0; persists to the GLIR flush (nothing else binds
    # GL_TEXTURE_BUFFER), and u_spikes samples unit 0. The one raw-GL touch gloo forces.
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

  def release(self):
    # Free the buffer texture on reload; guarded so cleanup can't break a reload.
    if self._tbo_tex is not None:
      try:
        gl.glDeleteTextures([self._tbo_tex])
      except Exception:
        pass
      self._tbo_tex = None

  # No __del__: glDeleteTextures from __del__ races interpreter shutdown (noisy PyOpenGL
  # message). release() (reload) or GL-context teardown frees the texture.


RasterHistory = create_visual_node(RasterHistoryVisual)


#### Full-history, zero-copy voltage traces ####
# Voltage analogue of the raster. Node writes voltage in place into a CUDA-registered GL
# buffer for the WHOLE run, (T, batch, n) float32 time-major (enable_voltage_history).
# Bound as a TEXTURE_BUFFER (R32F); vertex shader pulls v[t, neuron] via texelFetch to
# position each trace point. No per-frame copy, x = absolute time.
# #version 140: texelFetch + samplerBuffer in the vertex stage.
_VOLT_HIST_VERT = """
#version 140
attribute float a_time;     // absolute timestep for this vertex, 0..T-1
attribute float a_neuron;   // neuron id within the layer (which trace)
attribute vec4  a_color;    // per-neuron trace colour, static
uniform samplerBuffer u_volts;   // R32F history buffer; left UNSET -> texture unit 0
uniform int u_stride;            // floats per timestep row (= batch*n)
uniform float u_xoff;            // scroll offset (timesteps): shifts traces LEFT on
                                 // screen under a fixed camera. The texelFetch index
                                 // uses ABSOLUTE a_time, so only the x position moves.
varying vec4 v_color;
void main() {
    int idx = int(a_time) * u_stride + int(a_neuron);   // time-major; batch 0
    float v = texelFetch(u_volts, idx).r;
    gl_Position = $transform(vec4(a_time - u_xoff, v, 0.0, 1.0));
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

    # One vertex per (selected neuron, timestep); K*T total. Only static (time, neuron,
    # colour) are attributes; y (voltage) comes from the buffer texture.
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
    self.shared_program['u_xoff'] = 0.0      # no scroll until the widget drives it
    # u_volts never set: unset sampler -> unit 0, bound in _prepare_draw.

    # Static GL_LINES joining consecutive timesteps within each trace (built once).
    self._index_buffer = gloo.IndexBuffer(self._make_index())
    self._draw_mode = 'lines'
    self.set_gl_state('translucent', depth_test=False)

  def set_x_offset(self, x0):
    self.shared_program['u_xoff'] = float(x0)   # scroll via one uniform write

  def _make_index(self):
    # Edges (t, t+1) within each neuron's contiguous block of T vertices.
    T, K = self.T, self.K
    i = np.arange(T - 1)
    seg = np.stack([i, i + 1], axis=1)                       # (T-1, 2)
    return (seg[None] + (np.arange(K) * T)[:, None, None]).reshape(-1, 2).astype(np.uint32)

  def _create_tbo(self):
    # 1-D R32F view of the GL buffer; glTexBuffer references it (no copy).
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R32F, self._gl_buffer_id)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
    self._tbo_tex = tex

  def _prepare_draw(self, view):
    if self._tbo_tex is None:
      self._create_tbo()
    # Immediate raw-GL bind to unit 0; gloo's per-program flush keeps it live through
    # this draw (same as the raster, no collision on unit 0).
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self._tbo_tex)

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis == 0:
      return (0, self.T)
    return None   # y (voltage) bounds are data-dependent; the widget sets the camera

  def release(self):
    # Free the buffer texture on reload; guarded (see RasterHistoryVisual).
    if self._tbo_tex is not None:
      try:
        gl.glDeleteTextures([self._tbo_tex])
      except Exception:
        pass
      self._tbo_tex = None

  # No __del__ (see RasterHistoryVisual).


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
    self._texture_format = texture_format   # used by the CPU set_data fallback
    dummy = np.zeros((rows, cols), dtype=texture_format)
    # Explicit numeric clim (not 'auto'): 'auto' freezes on the first all-zero upload.
    super().__init__(data=dummy, texture_format=texture_format, clim=clim, cmap=cmap)
    self.freeze()

  def _register_texture(self):
    # Lazy: the gloo texture's GL object only exists after the first draw flushes it.
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
    # Push the current value matrix into the texture (a snapshot, no `t`).

    # CPU / no CUDA: no interop -- host-copy the whole matrix (cheap vs a CPU sim step).
    if driver is None or not self.value_getter().is_cuda:
      val = self.value_getter().detach().to('cpu').numpy().astype(
        self._texture_format, copy=False)
      self.set_data(val)
      self.update()
      return

    if self._cuda_tex_resource is None and not self._register_texture():
      return  # texture not on the GPU yet; skip this frame

    # Re-fetch each frame: learning rules may rebind feature.value to a new tensor.
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

  def release(self):
    # Unregister the CUDA-mapped texture on reload; guarded, idempotent.
    if self._cuda_tex_resource is not None and driver is not None:
      try:
        driver.cuGraphicsUnregisterResource(self._cuda_tex_resource)
      except Exception:
        pass
    self._cuda_tex_resource = None

  def __del__(self):
    self.release()


FeatureMatrix = create_visual_node(FeatureMatrixVisual)


#### Neurons-as-circles, firing read from the spike-history GL buffer ####
# One GL_POINTS vertex per neuron at a static layout position. Firing pulled zero-copy
# from the SAME R8UI spike-history buffer the raster reads: the vertex shader texelFetches
# this neuron's recent spikes for a fading "glow"; the frag shader draws a disc between
# base and fire colour. No per-frame copy: the widget just updates u_t.
# #version 140: texelFetch + usamplerBuffer.
_NEURON_VERT = """
#version 140
attribute vec2 a_pos;       // neuron position in DATA coords (static layout)
attribute float a_index;    // neuron id within the layer (row index into u_fire)
uniform usamplerBuffer u_fire;  // R8UI spike history; left UNSET -> texture unit 0
uniform int u_t;            // current timestep
uniform int u_T;            // total timesteps in the buffer
uniform int u_stride;       // elements per timestep row (= batch*n)
uniform int u_glow;         // afterglow window (timesteps); >=1
uniform float u_pointsize;  // on-screen disc diameter in pixels
varying float v_intensity;  // 0..1 firing glow -> fragment
void main() {
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
    gl_PointSize = u_pointsize;

    // Max spike over [t-u_glow+1, t] with linear falloff, so a spike stays visible
    // for a few frames even when draws are throttled (batch 0; time-major buffer).
    int idx = int(a_index);
    float inten = 0.0;
    for (int k = 0; k < u_glow; k++) {
        int tt = u_t - k;
        if (tt < 0 || tt >= u_T) continue;
        uint s = texelFetch(u_fire, tt * u_stride + idx).r;
        if (s != 0u) inten = max(inten, 1.0 - float(k) / float(u_glow));
    }
    v_intensity = inten;
}
"""

_NEURON_FRAG = """
#version 140
varying float v_intensity;
uniform vec4 u_base;        // resting colour
uniform vec4 u_fire_color;  // colour at full firing intensity
void main() {
    // Round the square point sprite into a disc.
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    gl_FragColor = mix(u_base, u_fire_color, v_intensity);
}
"""


class NeuronCloudVisual(Visual):
  def __init__(self, positions, indices, total_timesteps, row_stride, gl_buffer_id,
               base_color=(0.25, 0.25, 0.30, 1.0), fire_color=(1.0, 0.9, 0.2, 1.0),
               point_size=9.0, glow=8):
    self.n = int(len(positions))
    self.T = int(total_timesteps)
    self.stride = int(row_stride)            # = batch * n; idx = time*stride + neuron
    self._gl_buffer_id = int(gl_buffer_id)   # raw GL buffer with the spike history
    self._tbo_tex = None                     # GL texture viewing that buffer (lazy)
    Visual.__init__(self, vcode=_NEURON_VERT, fcode=_NEURON_FRAG)

    self._pos = np.asarray(positions, dtype=np.float32)
    self._pos_vbo = gloo.VertexBuffer(self._pos)
    self._index_vbo = gloo.VertexBuffer(np.asarray(indices, dtype=np.float32))
    self.shared_program['a_pos'] = self._pos_vbo
    self.shared_program['a_index'] = self._index_vbo
    self.shared_program['u_t'] = 0
    self.shared_program['u_T'] = self.T
    self.shared_program['u_stride'] = self.stride
    self.shared_program['u_glow'] = max(1, int(glow))
    self.shared_program['u_pointsize'] = float(point_size)
    self.shared_program['u_base'] = base_color
    self.shared_program['u_fire_color'] = fire_color
    # u_fire never set: unset sampler -> unit 0, bound in _prepare_draw.
    self._draw_mode = 'points'
    self.set_gl_state('translucent', depth_test=False)

  def set_time(self, t):
    self.shared_program['u_t'] = int(t)

  def _create_tbo(self):
    # 1-D R8UI view of the spike-history buffer; glTexBuffer references it (no copy).
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, tex)
    gl.glTexBuffer(gl.GL_TEXTURE_BUFFER, gl.GL_R8UI, self._gl_buffer_id)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, 0)
    self._tbo_tex = tex

  def _prepare_draw(self, view):
    if self._tbo_tex is None:
      self._create_tbo()
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)   # let the vertex shader's gl_PointSize apply
    # Immediate raw-GL bind to unit 0 (u_fire), same trick as the raster.
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_BUFFER, self._tbo_tex)

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis in (0, 1) and self.n:
      return (float(self._pos[:, axis].min()), float(self._pos[:, axis].max()))
    return None

  def release(self):
    # Free the buffer texture on reload; guarded (see RasterHistoryVisual).
    if self._tbo_tex is not None:
      try:
        gl.glDeleteTextures([self._tbo_tex])
      except Exception:
        pass
      self._tbo_tex = None

  # No __del__ (see RasterHistoryVisual).


NeuronCloud = create_visual_node(NeuronCloudVisual)


#### Synapses-as-lines ####
# A single GL_LINES draw covers every selected synapse across all connections: each
# segment is two vertices in `positions`, coloured per-vertex by weight (`colors`).
# Forward edges are one straight segment; recurrent/back edges are pre-tessellated into
# short segments along a bowed curve by the widget, so they share this one flat buffer.
# The colour buffer is rebuildable via set_colors (the weight-change rendering hook).
_SYNAPSE_VERT = """
#version 140
attribute vec2 a_pos;
attribute vec4 a_color;
varying vec4 v_color;
void main() {
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
    v_color = a_color;
}
"""

_SYNAPSE_FRAG = """
#version 140
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class SynapseLinesVisual(Visual):
  def __init__(self, positions, colors):
    Visual.__init__(self, vcode=_SYNAPSE_VERT, fcode=_SYNAPSE_FRAG)
    self._pos = np.asarray(positions, dtype=np.float32)
    self._pos_vbo = gloo.VertexBuffer(self._pos)
    self._color_vbo = gloo.VertexBuffer(np.asarray(colors, dtype=np.float32))
    self.shared_program['a_pos'] = self._pos_vbo
    self.shared_program['a_color'] = self._color_vbo
    self._draw_mode = 'lines'
    self.set_gl_state('translucent', depth_test=False)

  def set_colors(self, colors):
    # Weight-change hook. `colors` must match the construction vertex count.
    self._color_vbo.set_data(np.asarray(colors, dtype=np.float32))
    self.update()

  def _prepare_draw(self, view):
    pass

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis in (0, 1) and len(self._pos):
      return (float(self._pos[:, axis].min()), float(self._pos[:, axis].max()))
    return None


SynapseLines = create_visual_node(SynapseLinesVisual)


#### Cached synapse lines ####
# Synapse geometry is STATIC, but the canvas redraws every visual each frame, so plain
# SynapseLines re-pays its vertex-bound draw every frame (~24 ms, profiled). Instead: draw
# the lines ONCE into an offscreen FBO texture (over the data-space bbox), then draw a
# camera-transformed textured quad over that bbox each frame. The quad is in DATA coords,
# so the camera moves it like the lines -- the line pass only re-runs on a colour change
# (`set_colors`). Trade-off: a raster snapshot, so far zoom pixelates (raise `max_side`).
_CACHED_LINE_VERT = """
#version 120
attribute vec2 a_pos;
attribute vec4 a_color;
uniform vec2 u_scale;         // 2/(x1-x0), 2/(y1-y0): bbox -> clip, offscreen pass
uniform vec2 u_offset;        // (x0, y0)
varying vec4 v_color;
void main() {
    // Map the data-space bbox to clip space [-1, 1] directly (no matrix-convention
    // ambiguity): x0 -> -1, x1 -> +1, likewise y. The FBO viewport then puts x0,y0 at
    // texel (0, 0), matching the display quad's (0, 0) texcoord at corner (x0, y0).
    vec2 ndc = (a_pos - u_offset) * u_scale - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = a_color;
}
"""

_CACHED_LINE_FRAG = """
#version 120
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""

# Display quad: a textured rectangle spanning the bbox in data coords, positioned by
# the scene transform (camera). vispy's default GLSL handles attribute/varying and the
# $transform Function injection.
_CACHED_QUAD_VERT = """
attribute vec2 a_pos;
attribute vec2 a_tex;
varying vec2 v_tex;
void main() {
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));
    v_tex = a_tex;
}
"""

_CACHED_QUAD_FRAG = """
uniform sampler2D u_tex;
varying vec2 v_tex;
void main() { gl_FragColor = texture2D(u_tex, v_tex); }
"""


class CachedSynapseLinesVisual(Visual):
  def __init__(self, positions, colors, bbox, max_side=2048):
    Visual.__init__(self, vcode=_CACHED_QUAD_VERT, fcode=_CACHED_QUAD_FRAG)
    x0, y0, x1, y1 = (float(v) for v in bbox)
    # Guard against a degenerate (zero-area) bbox.
    if x1 <= x0:
      x1 = x0 + 1.0
    if y1 <= y0:
      y1 = y0 + 1.0
    self._bbox = (x0, y0, x1, y1)

    # Offscreen resolution: longest side = max_side, other side by aspect.
    aspect = (x1 - x0) / (y1 - y0)
    if aspect >= 1.0:
      W, H = int(max_side), max(16, int(round(max_side / aspect)))
    else:
      W, H = max(16, int(round(max_side * aspect)), ), int(max_side)
    self._W, self._H = int(W), int(H)

    # Offscreen line program (raw gloo; rendered into the FBO in refresh()).
    self._line_prog = gloo.Program(_CACHED_LINE_VERT, _CACHED_LINE_FRAG)
    self._line_prog['a_pos'] = gloo.VertexBuffer(np.asarray(positions, dtype=np.float32))
    self._line_color = gloo.VertexBuffer(np.asarray(colors, dtype=np.float32))
    self._line_prog['a_color'] = self._line_color
    self._line_prog['u_scale'] = (2.0 / (x1 - x0), 2.0 / (y1 - y0))
    self._line_prog['u_offset'] = (x0, y0)
    self._n_verts = int(len(positions))

    self._tex = None     # FBO colour texture (lazy; needs a GL context)
    self._fbo = None
    self.dirty = True    # needs an offscreen render before the quad is meaningful

    # Display quad over the bbox (data coords) with matching texcoords.
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float32)
    texco = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    self.shared_program['a_pos'] = gloo.VertexBuffer(corners)
    self.shared_program['a_tex'] = gloo.VertexBuffer(texco)
    self._draw_mode = 'triangle_strip'
    self.set_gl_state('translucent', depth_test=False)

  def refresh(self):
    # Render the static lines into the offscreen texture. Runs OUTSIDE the scene draw
    # (from the widget's render()) so the nested FBO pass doesn't interleave with GLIR.
    if self._tex is None:
      self._tex = gloo.Texture2D(
        shape=(self._H, self._W, 4), format='rgba', interpolation='linear')
      self._fbo = gloo.FrameBuffer(color=self._tex)
      self.shared_program['u_tex'] = self._tex

    with self._fbo:
      gloo.set_viewport(0, 0, self._W, self._H)
      gloo.set_state(blend=True, depth_test=False,
                     blend_func=('src_alpha', 'one_minus_src_alpha'))
      gloo.clear(color=(0.0, 0.0, 0.0, 0.0))
      self._line_prog.draw('lines')
    # FrameBuffer.__exit__ doesn't restore the viewport, so reset it to the full canvas
    # (else the next on-screen draw is squashed into the FBO's (W, H) rect).
    canvas = getattr(self, 'canvas', None)
    if canvas is not None:
      w, h = canvas.physical_size
      gloo.set_viewport(0, 0, int(w), int(h))
    self.dirty = False

  def set_colors(self, colors):
    # Weight-change hook: update colours, mark stale so the next render() re-bakes.
    self._line_color.set_data(np.asarray(colors, dtype=np.float32))
    self.dirty = True

  def _prepare_draw(self, view):
    if self._tex is None:
      self.refresh()   # first frame: bake so the quad has something to sample

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if axis == 0:
      return (self._bbox[0], self._bbox[2])
    if axis == 1:
      return (self._bbox[1], self._bbox[3])
    return None

  def release(self):
    # Drop the FBO + colour texture (gloo objects are GC-freed) on reload.
    self._fbo = None
    self._tex = None


CachedSynapseLines = create_visual_node(CachedSynapseLinesVisual)


#### Scrolling "oscilloscope" time axis ####
# Plots scroll a trailing window under a PINNED camera via one uniform (u_xoff) -- a
# camera move fires the transform cascade + a per-draw axis glyph/VBO re-upload that
# ~halves steps/s. These two visuals are the axis analogue: ticks + labels for the whole
# timeline built ONCE, scrolled by the same u_xoff (one scalar write/draw, no re-layout).
# They sit in a gutter ViewBox below the plot with matching x-range, so labels line up
# with the data. Shown only while running; the vispy AxisWidget takes over on pause.

### Tick labels that scroll via a uniform ###
# Subclasses TextVisual, swapping only the vertex shader to subtract u_xoff before
# $transform. Anchor positions never change, so TextVisual's per-glyph VBO re-upload
# (`_pos_changed`) never runs after the build. NOTE: a transform change re-trips it, so
# the gutter camera must stay pinned (it is).
_SCROLL_TEXT_VERT = _TEXT_VERT.replace(
    "attribute vec3 a_pos;  // anchor position",
    "attribute vec3 a_pos;  // anchor position\n"
    "    uniform float u_xoff;  // scroll offset (timesteps): shifts labels LEFT",
).replace(
    "$transform(vec4(a_pos, 1.0))",
    "$transform(vec4(a_pos.x - u_xoff, a_pos.y, a_pos.z, 1.0))",
)
# Fail loud if a vispy upgrade changes the shader out from under the patch.
assert "u_xoff" in _SCROLL_TEXT_VERT and "a_pos.x - u_xoff" in _SCROLL_TEXT_VERT, \
    "vispy TextVisual vertex shader changed; update _SCROLL_TEXT_VERT patch"


class ScrollingLabelsVisual(TextVisual):
  _shaders = {'vertex': _SCROLL_TEXT_VERT, 'fragment': _TEXT_FRAG}

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.shared_program['u_xoff'] = 0.0   # no scroll until the widget drives it

  def set_x_offset(self, x0):
    self.shared_program['u_xoff'] = float(x0)


ScrollingLabels = create_visual_node(ScrollingLabelsVisual)


### Tick marks + static axis baseline ###
# One GL_LINES draw scrolled by u_xoff, built once. Per-vertex colour draws white
# baseline + grey ticks in a single call, matching the stock AxisVisual.
_MARKS_VERT = """
attribute vec2 a_pos;     // (time, gutter_y); gutter_y in [0,1], top (axis line)=1
attribute vec4 a_color;   // per-vertex colour (white baseline, grey ticks)
uniform float u_xoff;     // scroll offset (timesteps): shifts marks LEFT on screen
varying vec4 v_color;
void main() {
    gl_Position = $transform(vec4(a_pos.x - u_xoff, a_pos.y, 0.0, 1.0));
    v_color = a_color;
}
"""

_MARKS_FRAG = """
varying vec4 v_color;
void main() { gl_FragColor = v_color; }
"""


class ScrollingMarksVisual(Visual):
  def __init__(self, positions, colors):
    Visual.__init__(self, vcode=_MARKS_VERT, fcode=_MARKS_FRAG)
    self._pos = np.asarray(positions, dtype=np.float32)
    self._pos_vbo = gloo.VertexBuffer(self._pos)
    self._color_vbo = gloo.VertexBuffer(np.asarray(colors, dtype=np.float32))
    self.shared_program['a_pos'] = self._pos_vbo
    self.shared_program['a_color'] = self._color_vbo
    self.shared_program['u_xoff'] = 0.0
    self._draw_mode = 'lines'
    self.set_gl_state('translucent', depth_test=False)

  def set_x_offset(self, x0):
    self.shared_program['u_xoff'] = float(x0)

  def _prepare_draw(self, view):
    pass

  def _prepare_transforms(self, view):
    view.view_program.vert['transform'] = view.transforms.get_transform()

  def _compute_bounds(self, axis, view):
    if len(self._pos) and axis in (0, 1):
      return (float(self._pos[:, axis].min()), float(self._pos[:, axis].max()))
    return None


ScrollingMarks = create_visual_node(ScrollingMarksVisual)


#### Cached static chrome (axes / ticks / labels / titles) ####
# The canvas re-processes every AxisVisual/TextVisual on the CPU each draw, though the
# "chrome" is identical frame to frame (~10 ms of a ~14 ms draw). ChromeCache bakes it to
# one texture and draws it back as a fullscreen quad each frame; THIS visual is that quad.
# A NullTransform makes a_pos (already clip-space) bypass the camera. Transparent where
# the live plots show through.
_CHROME_OVL_VERT = """
attribute vec2 a_pos;     // fullscreen-quad corner in CLIP space [-1, 1]
attribute vec2 a_tex;     // matching texcoord [0, 1]
varying vec2 v_tex;
void main() {
    gl_Position = $transform(vec4(a_pos, 0.0, 1.0));   // NullTransform -> identity
    v_tex = a_tex;
}
"""

_CHROME_OVL_FRAG = """
uniform sampler2D u_tex;   // baked chrome (RGBA; transparent over the plot regions)
varying vec2 v_tex;
void main() { gl_FragColor = texture2D(u_tex, v_tex); }
"""


class ChromeOverlayVisual(Visual):
  def __init__(self):
    Visual.__init__(self, vcode=_CHROME_OVL_VERT, fcode=_CHROME_OVL_FRAG)
    self._pos_vbo = gloo.VertexBuffer(
        np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32))
    self._tex_vbo = gloo.VertexBuffer(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32))
    self.shared_program['a_pos'] = self._pos_vbo
    self.shared_program['a_tex'] = self._tex_vbo
    self._draw_mode = 'triangle_strip'
    self.set_gl_state('translucent', depth_test=False)

  def set_texture(self, texture):
    self.shared_program['u_tex'] = texture

  def _prepare_draw(self, view):
    return True

  def _prepare_transforms(self, view):
    # a_pos is already clip-space -> bypass the camera, fill the whole framebuffer.
    view.view_program.vert['transform'] = NullTransform()


ChromeOverlay = create_visual_node(ChromeOverlayVisual)
