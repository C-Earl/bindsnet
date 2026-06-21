from vispy import scene
import numpy as np
from abc import abstractmethod
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram


class AbstractWidget:
  def __init__(self):
    # self.view = scene.widgets.ViewBox()   # VisPy ViewBox for widget rendering
    self.grid = scene.widgets.Grid()    # Grid to hold widget view and axes (if applicable)
    self.history = []       # List to store historical data for rendering. One element per time-step

  @abstractmethod
  def prime(self, network):
    pass

  @abstractmethod
  def render(self):
    pass

  @abstractmethod
  def get_history(self):
    pass

  def reset(self):
    self.history = []


# A plotting widget with x and y axis
class GraphPlotWidget(AbstractWidget):
  def __init__(self, ):
    super().__init__()

    ### Create plot with x and y axes ###
    self.view = self.grid.add_view(
      row=0,
      col=0,
      border_color='white'
    )
    self.view.camera = 'panzoom'
    # self.x_axis = scene.AxisWidget(
    #   orientation='bottom'
    # )
    # self.y_axis = scene.AxisWidget(
    #   orientation='left'
    # )
    # self.grid.add_widget(
    #   self.y_axis,
    #   row=0,
    #   col=0
    # )
    # self.grid.add_widget(
    #   self.x_axis,
    #   row=0,
    #   col=0
    # )
    # self.x_axis.link_view(self.view)
    # self.y_axis.link_view(self.view)


class RasterPlot(GraphPlotWidget):
  def __init__(self,
               layer_name: str,
               max_timesteps: int = 100):
    super().__init__()
    self.layer_name = layer_name
    self.layer = None           # Initialized after added to Application object
    self.max_timesteps = max_timesteps

    # self.view.camera = 'panzoom'
    self.markers = scene.visuals.Markers(
      parent=self.view.scene
    )

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n

  def render(self, t):
    ### Extract spike data from layer ###
    spike_data = self.layer.s.cpu().numpy()
    spike_ids = np.where(spike_data > 0)[1]
    for sid in spike_ids:
        self.history.append([t, sid])

    if len(self.history) == 0:
        return

    ### Render ###
    points = np.array(self.history, dtype=np.float32)
    self.markers.set_data(
        points,
        face_color='white',
        size=4
    )
    self.view.camera.set_range(
        x=(max(0, t - self.max_timesteps), max(self.max_timesteps, t)),
        y=(0, self.layer_size)
    )

  def get_history(self):
    return np.array(self.history, dtype=np.float32)


class VoltagePlot(AbstractWidget):
  def __init__(self,
            width: float,
            height: float,
            x: float,
            y: float,
            layer_name: str,
            neuron_ids: list[int],
            max_timesteps: int = 100,
            y_range: tuple[float, float] = (-80.0, 40.0)
  ):

    super().__init__(width, height, x, y)
    self.layer_name = layer_name
    self.layer = None
    self.max_timesteps = max_timesteps
    self.neuron_ids = neuron_ids
    self.history = {}   # Dictionary mapping neuron ID to list of [timestep, voltage] pairs
    self.view.camera = 'panzoom'
    self.lines = {}
    self.y_range = y_range  # Plotted y-axis range

    # Initial camera range
    self.view.camera.set_range(
      x=(0, self.max_timesteps),
      y=(self.y_range[0], self.y_range[1])  # Typical membrane voltage range
    )

  def prime(self, network):

    self.layer = network.layers[self.layer_name]
    for nid in self.neuron_ids:
      self.history[nid] = []
      self.lines[nid] = scene.visuals.Line(
        parent=self.view.scene,
        width=2
      )

  def render(self, t):
    ### Extract voltage data from layer ###
    voltages = self.layer.v
    voltages = voltages.cpu().numpy().flatten()   # TODO: Make this more efficient/GPU-friendly?
    for nid in self.neuron_ids:
      v = voltages[nid]
      self.history[nid].append([t, v])

    all_values = []
    for nid in self.neuron_ids:
      points = np.array(
        self.history[nid],
        dtype=np.float32
      )

      if len(points) < 2:
        continue

      self.lines[nid].set_data(points)
      all_values.extend(points[:, 1])

    ### Render ###
    xmin = max(0, t - self.max_timesteps)
    xmax = max(self.max_timesteps, t)

    # Autoscale voltage range
    if len(all_values) > 0:
      ymin = min(all_values)
      ymax = max(all_values)
      padding = max(1.0, (ymax - ymin) * 0.1)
      self.view.camera.set_range(
        x=(xmin, xmax),
        y=(ymin - padding, ymax + padding)
      )

  def get_history(self, neuron_id):
    return np.array(
      self.history[neuron_id],
      dtype=np.float32
    )


from .visuals import RasterTexture
class AdvancedRasterPlot(GraphPlotWidget):
  def __init__(self,
               layer_name: str,
               max_timesteps: int = 100):
    super().__init__()
    self.layer_name = layer_name
    self.layer = None           # Initialized after added to Application object
    self.spikes_vbo = None      # Initialized after added to Application object
    self.max_timesteps = max_timesteps
    self.current_write_head = 0
    self.raster = None

  def prime(self, network):
    self.layer = network.layers[self.layer_name]
    self.layer_size = self.layer.n
    self.spikes_vbo = network.opengl_vbos['layers'][self.layer_name]['s']
    self.raster = RasterTexture(
      layer_size=self.layer_size,
      width=self.max_timesteps,
      spike_vbo=self.spikes_vbo
    )
    self.view.add(self.raster)
    self.view.camera.rect = (0, 0, self.max_timesteps, self.layer_size)


  def render(self, t):
    self.raster.migrate_spikes(t)


  def get_history(self):
    return np.array(self.history, dtype=np.float32)

# class AdvancedRasterPlot(GraphPlotWidget):
#   def __init__(self,
#                layer_name: str,
#                max_timesteps: int = 100):
#     super().__init__()
#     self.layer_name = layer_name
#     self.layer = None           # Initialized after added to Application object
#     self.spikes_vbo = None      # Initialized after added to Application object
#     self.max_timesteps = max_timesteps
#     self.current_write_head = 0
#     # self.markers = scene.visuals.Markers(
#     #   parent=self.view.scene
#     # )
#
#     ### OpenGL Rendering program ###
#     ### Define raster shaders ###
#     raster_texture_vertex_shader = """
#             #version 330 core
#
#             layout(location = 0) in vec2 pos;
#             out vec2 uv;
#             void main()
#             {
#                 uv = pos * 0.5 + 0.5;
#                 gl_Position = vec4(pos, 0.0, 1.0);
#             }
#             """
#     raster_texture_fragment_shader = """
#             #version 330 core
#
#             in vec2 uv;
#             out vec4 FragColor;
#             uniform sampler2D raster_tex;
#             uniform float write_head;
#             uniform float history_width;
#
#             void main()
#               {
#                   FragColor = vec4(1,0,0,1);
#               }
#         """
#     self.raster_plot_program = compileProgram(
#       compileShader(raster_texture_vertex_shader, gl.GL_VERTEX_SHADER),
#       compileShader(raster_texture_fragment_shader, gl.GL_FRAGMENT_SHADER)
#     )
#
#     ### Vertex indices buffer (square covering widget) ###
#     quad_vertices = np.array([
#       -1.0, -1.0,
#       1.0, -1.0,
#       1.0, 1.0,
#
#       -1.0, -1.0,
#       1.0, 1.0,
#       -1.0, 1.0,
#     ], dtype=np.float32)
#     self.quad_vao = gl.glGenVertexArrays(1)
#     quad_vbo = gl.glGenBuffers(1)
#     gl.glBindVertexArray(self.quad_vao)
#     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vbo)
#     gl.glBufferData(
#       gl.GL_ARRAY_BUFFER,  # Target buffer
#       quad_vertices.nbytes,  # Size of data in bytes
#       quad_vertices,  # Data
#       gl.GL_STATIC_DRAW  # Type of drawing (static data, not changing)
#     )
#     gl.glVertexAttribPointer(
#       0,  # VAO slot
#       2,  # x,y
#       gl.GL_FLOAT,  # Data type
#       False,  # Normalized?
#       0,  # Stride
#       None  # Offset in buffer
#     )
#     gl.glEnableVertexAttribArray(0)
#     gl.glBindVertexArray(0)
#
#   def prime(self, network):
#     self.layer = network.layers[self.layer_name]
#     self.layer_size = self.layer.n
#     self.spikes_vbo = network.opengl_vbos['layers'][self.layer_name]['s']
#
#     ### Define texture for rolling spike buffer ###
#     self.raster_texture = gl.glGenTextures(1)
#     gl.glBindTexture(gl.GL_TEXTURE_2D, self.raster_texture)
#     gl.glTexImage2D(
#       gl.GL_TEXTURE_2D,
#       0,  # Mipmap level
#       gl.GL_R8,  # Internal format (32-bit float)
#       self.max_timesteps,  # Width of texture (time steps)
#       self.layer_size,  # Height of texture (neurons)
#       0,  # Border
#       gl.GL_RED,  # Format of pixel data
#       gl.GL_UNSIGNED_BYTE,  # Data type of pixel data
#       np.zeros(
#         (self.layer_size, self.max_timesteps),
#         dtype=np.uint8
#       )  # No initial data
#     )
#     gl.glTexParameteri(
#       gl.GL_TEXTURE_2D,
#       gl.GL_TEXTURE_MIN_FILTER,
#       gl.GL_NEAREST
#     )
#
#   def render(self, t):
#     # ### Extract spike data from layer ###
#     # spike_data = self.layer.s.cpu().numpy()
#     # spike_ids = np.where(spike_data > 0)[1]
#     # for sid in spike_ids:
#     #   self.history.append([t, sid])
#     #
#     # if len(self.history) == 0:
#     #   return
#     #
#     # ### Render ###
#     # points = np.array(self.history, dtype=np.float32)
#     # self.markers.set_data(
#     #   points,
#     #   face_color='white',
#     #   size=4
#     # )
#     # self.view.camera.set_range(
#     #   x=(max(0, t - self.max_timesteps), max(self.max_timesteps, t)),
#     #   y=(0, self.layer_size)
#     # )
#
#     wrapped_t = t % self.max_timesteps
#
#     gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
#
#     gl.glBindBuffer(
#       gl.GL_PIXEL_UNPACK_BUFFER,
#       self.spikes_vbo
#     )
#
#     gl.glBindTexture(
#       gl.GL_TEXTURE_2D,
#       self.raster_texture
#     )
#
#     gl.glTexSubImage2D(
#       gl.GL_TEXTURE_2D,
#       0,
#       wrapped_t,
#       0,
#       1,
#       self.layer_size,
#       gl.GL_RED,
#       gl.GL_UNSIGNED_BYTE,
#       None
#     )
#
#     self.current_write_head = wrapped_t
#
#     self.view.camera.set_range(
#       x=(max(0, t - self.max_timesteps),
#          max(self.max_timesteps, t)),
#       y=(0, self.layer_size)
#     )
#
#   def draw(self):
#     print("DRAW")
#     # rect = self.view.rect
#
#     # canvas_h = self.canvas.physical_size[1]
#     #
#     # x = int(rect.left)
#     # y = int(canvas_h - rect.bottom - rect.height)
#     #
#     # w = int(rect.width)
#     # h = int(rect.height)
#     prev_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
#     #
#     # gl.glViewport(x, y, w, h)
#     #
#     # gl.glEnable(gl.GL_SCISSOR_TEST)
#     # gl.glScissor(x, y, w, h)
#
#     vp = *self.view.rect.pos, *(int(i) for i in self.view.rect.size)
#     # gl.glViewport(*vp)
#     # gl.glEnable(gl.GL_SCISSOR_TEST)
#     # gl.glScissor(*vp)
#
#     gl.glUseProgram(self.raster_plot_program)
#
#     gl.glUniform1f(
#       gl.glGetUniformLocation(
#         self.raster_plot_program,
#         "write_head"
#       ),
#       self.current_write_head
#     )
#
#     gl.glUniform1f(
#       gl.glGetUniformLocation(
#         self.raster_plot_program,
#         "history_width"
#       ),
#       self.max_timesteps
#     )
#
#     tex_loc = gl.glGetUniformLocation(
#       self.raster_plot_program,
#       "raster_tex"
#     )
#
#     gl.glUniform1i(tex_loc, 0)
#
#     gl.glActiveTexture(gl.GL_TEXTURE0)
#     gl.glBindTexture(
#       gl.GL_TEXTURE_2D,
#       self.raster_texture
#     )
#
#     gl.glBindVertexArray(self.quad_vao)
#
#     gl.glDrawArrays(
#       gl.GL_TRIANGLES,
#       0,
#       6
#     )
#
#     ### Unbind everything ###
#     gl.glUseProgram(0)
#     gl.glBindVertexArray(0)
#     gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
#     gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
#     gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
#     gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
#     gl.glDisable(gl.GL_SCISSOR_TEST)
#
#   def get_history(self):
#     return np.array(self.history, dtype=np.float32)