import torch
import glfw
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np

from bindsnet.network.network import GUINetwork


class AbstractWidget:
  def __init__(self, width: float, height: float, x:float, y:float):
    self.width = width      # Widget width
    self.height = height    # Widget height
    self.x = x              # Bottom-left x coordinate
    self.y = y              # Bottom-right y coordinate

    ### Widget border rendering ###
    vertices = np.array([
      -0.99, -0.99,
      0.99, -0.99,
      0.99, 0.99,
      -0.99, 0.99,
    ], dtype=np.float32)

    ### Generate VAO for border geometry ###
    self.widget_border_vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(self.widget_border_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(
      gl.GL_ARRAY_BUFFER,   # Target buffer
      vertices.nbytes,      # Size of data in bytes
      vertices,             # Data
      gl.GL_STATIC_DRAW     # Type of drawing (static data, not changing frequently)
    )
    gl.glVertexAttribPointer(
      0,            # VAO slot
      2,            # x,y
      gl.GL_FLOAT,  # Data type
      False,        # Normalized?
      0,            # Stride
      None          # Offset in buffer
    )
    gl.glEnableVertexAttribArray(0)
    gl.glBindVertexArray(0)
    widget_border_vertex_shader = """
      #version 330 core
      layout(location = 0) in vec2 pos;
      void main()
      {
          gl_Position = vec4(pos, 0.0, 1.0);
      }
    """
    widget_border_fragment_shader = """
      #version 330 core
      out vec4 FragColor;
      void main()
      {
          FragColor = vec4(1.0, 1.0, 1.0, 1.0);
      }
    """
    self.border_line_shader = compileProgram(
      compileShader(widget_border_vertex_shader, gl.GL_VERTEX_SHADER),
      compileShader(widget_border_fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

  def set_window(self, app_window: glfw._GLFWwindow):
    self.window = app_window

  def render_widget_border(self):
    gl.glViewport(self.x, self.y, self.width, self.height)
    gl.glUseProgram(self.border_line_shader)
    gl.glBindVertexArray(self.widget_border_vao)
    gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 4)
    gl.glBindVertexArray(0)

  def render(self, time_step: int):
    pass

class RasterPlotWidget(AbstractWidget):
  # language=rst
  """
  Render a raster plot

  :param width: Width of the raster plot
  :param height: Height of the raster plot
  :param vao: Vertex Array Object index containing spike data
  :param layer_size: Number of neurons in the layer being plotted
  :return: None
  """
  def __init__(self,
      width: float,
      height: float,
      x:float,
      y:float,
      layer_name: int,
      tick_spacing: int=200,
    ):
    super().__init__(width, height, x, y)
    self.layer_name = layer_name
    self.max_time_steps = width
    self.window = None        # Assigned when App.add_widget() called
    self.spikes_vbo = None    # Assigned when App.add_widget() called
    self.layer_size = None    # Assigned when App.add_widget() called
    self.tick_spacing = tick_spacing
    self.layer = None
    border_inset = min(width*0.1, height*0.1)  # padding from edges of widget to border
    self.drawable_width = int(self.width - 2*border_inset)
    self.drawable_height = int(self.height - 2*border_inset)
    self.drawable_x = int(self.x + border_inset)  # Leave room on right for tick/axis labels
    self.drawable_y = int(self.y + border_inset)  # Leave room on bottom for tick/axis labels
    self.x_tick_width = self.drawable_width
    self.x_tick_height = int(height*0.05)
    self.x_tick_x = self.drawable_x
    self.x_tick_y = self.drawable_y - self.x_tick_height

    ### Define raster shaders ###
    raster_texture_vertex_shader = """
        #version 330 core

        layout(location = 0) in vec2 pos;
        out vec2 uv;
        void main()
        {
            uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
    raster_texture_fragment_shader = """
        #version 330 core

        in vec2 uv;
        out vec4 FragColor;
        uniform sampler2D raster_tex;
        uniform float write_head;
        uniform float history_width;

        void main()
        {   
            int x = int(uv.x * history_width);
            int y = int(uv.y * history_width);

            int shifted_x = int(
              mod((history_width + x + 1) + write_head, history_width)
            );

            ivec2 texel_coord = ivec2(shifted_x, y);
            float spike =
                texelFetch(
                    raster_tex,
                    texel_coord,
                    0
                ).r;
            vec3 color = vec3(spike*255);
            FragColor = vec4(color, 1.0);
        }
    """
    self.raster_plot_program = compileProgram(
      compileShader(raster_texture_vertex_shader, gl.GL_VERTEX_SHADER),
      compileShader(raster_texture_fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

    ### Vertex indices buffer (square covering widget) ###
    quad_vertices = np.array([
      -1.0, -1.0,
      1.0, -1.0,
      1.0, 1.0,

      -1.0, -1.0,
      1.0, 1.0,
      -1.0, 1.0,
    ], dtype=np.float32)
    self.quad_vao = gl.glGenVertexArrays(1)
    quad_vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(self.quad_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vbo)
    gl.glBufferData(
      gl.GL_ARRAY_BUFFER,     # Target buffer
      quad_vertices.nbytes,   # Size of data in bytes
      quad_vertices,          # Data
      gl.GL_STATIC_DRAW       # Type of drawing (static data, not changing)
    )
    gl.glVertexAttribPointer(
      0,            # VAO slot
      2,            # x,y
      gl.GL_FLOAT,  # Data type
      False,        # Normalized?
      0,            # Stride
      None          # Offset in buffer
    )
    gl.glEnableVertexAttribArray(0)
    gl.glBindVertexArray(0)

    ### Define tick shaders ###
    tick_vertex_shader = """
    #version 330 core

    layout(location = 0) in vec2 pos;
    void main()
    {
        gl_Position = vec4(pos, 0.0, 1.0);
    }
    """

    tick_fragment_shader = """
    #version 330 core

    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    """
    self.tick_program = compileProgram(
      compileShader(tick_vertex_shader, gl.GL_VERTEX_SHADER),
      compileShader(tick_fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

    ### Define tick VAO/VBO ###
    self.tick_vao = gl.glGenVertexArrays(1)
    self.tick_vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(self.tick_vao)
    gl.glBindBuffer(
      gl.GL_ARRAY_BUFFER,
      self.tick_vbo
    )
    gl.glBufferData(
      gl.GL_ARRAY_BUFFER,
      1024 * 1024,    # TODO: Set to max number of possible ticks
      None,
      gl.GL_DYNAMIC_DRAW
    )
    gl.glVertexAttribPointer(
      0,            # VAO slot
      2,            # x,y
      gl.GL_FLOAT,  # Data type
      False,        # Normalized?
      0,            # Stride
      None          # Offset in buffer
    )
    gl.glEnableVertexAttribArray(0)
    gl.glBindVertexArray(0)

  def prime(self, network: GUINetwork):
    self.layer_size = network.layers[self.layer_name].n
    self.spikes_vbo = network.opengl_vbos['layers'][self.layer_name]['s']

    ### Define texture for rolling spike buffer ###
    self.raster_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.raster_texture)
    gl.glTexImage2D(
      gl.GL_TEXTURE_2D,
      0,                    # Mipmap level
      gl.GL_R8,             # Internal format (32-bit float)
      self.max_time_steps,  # Width of texture (time steps)
      self.layer_size,      # Height of texture (neurons)
      0,                    # Border
      gl.GL_RED,            # Format of pixel data
      gl.GL_UNSIGNED_BYTE,  # Data type of pixel data
      np.zeros(
        (self.layer_size, self.max_time_steps),
        dtype=np.uint8
      )  # No initial data
    )
    gl.glTexParameteri(
      gl.GL_TEXTURE_2D,
      gl.GL_TEXTURE_MIN_FILTER,
      gl.GL_NEAREST
    )
    gl.glPixelStorei(
      gl.GL_UNPACK_ALIGNMENT,
      1
    )

    self.layer = network.layers[self.layer_name]

  def render_ticks(self, time_step: int) -> None:
    # Set size of area we are rendering into
    gl.glViewport(self.x_tick_x, self.x_tick_y, self.x_tick_width, self.x_tick_height)

    gl.glEnable(gl.GL_SCISSOR_TEST)

    gl.glScissor(
      self.x_tick_x,
      self.x_tick_y,
      self.x_tick_width,
      self.x_tick_height
    )

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    gl.glDisable(gl.GL_SCISSOR_TEST)

    ### Calculate tick labels and position vertices ###
    t_s = time_step - self.max_time_steps    # Oldest time step currently visible in raster plot
    # Labels
    label_range = (
      max(t_s + (self.tick_spacing - (t_s % self.tick_spacing)), 0),
      time_step - (time_step % self.tick_spacing)
    )
    labels = np.arange(
      label_range[0],
      label_range[1] + 1, self.tick_spacing
    )
    # Tick positions
    tick_x_pos = (
        (labels - t_s)
        / self.max_time_steps
    )
    tick_x_pos = (
      tick_x_pos * 2.0
    ) - 1.0
    # tick_x_pos = (((tick_x_pos - t_s) / self.max_time_steps) * 2) - 1  # Normalize to [-1,1] for shader
    # Vertices
    y_top = 1.0
    y_bot = 0.3
    vertices = np.array(
    [
        [x, y_top,
         x, y_bot]
        for x in tick_x_pos
    ], dtype=np.float32
    ).flatten()

    ### Render ###
    gl.glBindBuffer(
      gl.GL_ARRAY_BUFFER,
      self.tick_vbo
    )
    gl.glBufferSubData(
      gl.GL_ARRAY_BUFFER,
      0,
      vertices.nbytes,
      vertices
    )
    gl.glUseProgram(self.tick_program)
    gl.glBindVertexArray(self.tick_vao)
    gl.glDrawArrays(gl.GL_LINES, 0, len(vertices)//2)
    gl.glBindVertexArray(0)

  def render_spikes(self, time_step: int) -> None:
    # Set size of area we are rendering into
    gl.glViewport(self.drawable_x, self.drawable_y, self.drawable_width, self.drawable_height)

    wrapped_t = time_step % self.max_time_steps

    ### Migrate spike data to GPU ###
    gl.glBindBuffer(
      gl.GL_PIXEL_UNPACK_BUFFER,
      self.spikes_vbo
    )
    gl.glBindTexture(gl.GL_TEXTURE_2D,
                     self.raster_texture)
    gl.glTexSubImage2D(
      gl.GL_TEXTURE_2D,
      0,
      wrapped_t,  # x offset
      0,          # y offset
      1,          # width
      self.layer_size,  # height
      gl.GL_RED,
      gl.GL_UNSIGNED_BYTE,
      None
    )

    ### Plot ###
    # Pass write head and length of history to shader
    gl.glUseProgram(self.raster_plot_program)
    gl.glUniform1f(
      gl.glGetUniformLocation(self.raster_plot_program, "write_head"),
      wrapped_t
    )
    gl.glUniform1f(   # TODO: Can this be manually added into shader string definition?
      gl.glGetUniformLocation(self.raster_plot_program, "history_width"),
      self.max_time_steps
    )

    # Draw texture (spikes)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(
      gl.GL_TEXTURE_2D,
      self.raster_texture
    )
    gl.glBindVertexArray(self.quad_vao)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    glfw.swap_buffers(self.window)
    glfw.poll_events()

  def render(self, time_step: int):
    super().render(time_step)
    # self.render_background()
    self.render_ticks(time_step)
    self.render_spikes(time_step)
