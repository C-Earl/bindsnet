import tempfile
from typing import Dict, Iterable, Optional, Type, Any

import torch
from numpy import dtype
from torch import Tensor

from bindsnet.learning.reward import AbstractReward
from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import CSRMNodes, Nodes, Input, LIFNodes
from bindsnet.network.topology import AbstractConnection, AbstractMulticompartmentConnection


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from
        disk.
    """
    network = torch.load(
        open(file_name, "rb"), map_location=map_location, weights_only=False
    )
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Central object of the ``bindsnet`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
        self, connection: AbstractConnection | AbstractMulticompartmentConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.serialization.add_safe_globals([self])
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.

        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    if isinstance(target, CSRMNodes):
                        inputs[c[1]] = torch.zeros(
                            self.batch_size,
                            target.res_window_size,
                            *target.shape,
                            device=target.s.device,
                        )
                    else:
                        inputs[c[1]] = torch.zeros(
                            self.batch_size, *target.shape, device=target.s.device
                        )

                # Add to input: source's spikes multiplied by connection weights.
                if isinstance(target, CSRMNodes):
                    inputs[c[1]] += self.connections[c].compute_window(source.s)
                else:
                    inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        :param Bool progress_bar: Show a progress bar while running the network.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Run synapse updates.
        if "a_minus" in kwargs:
            A_Minus = kwargs["a_minus"]
            kwargs.pop("a_minus")
            if isinstance(A_Minus, dict):
                A_MD = True
            else:
                A_MD = False
        else:
            A_Minus = None

        if "a_plus" in kwargs:
            A_Plus = kwargs["a_plus"]
            kwargs.pop("a_plus")
            if isinstance(A_Plus, dict):
                A_PD = True
            else:
                A_PD = False
        else:
            A_Plus = None

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(
                        x=torch.zeros(
                            self.layers[l].s.shape, device=self.layers[l].s.device
                        )
                    )

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

            for c in self.connections:
                flad_m = False
                if A_Minus != None and ((isinstance(A_Minus, float)) or (c in A_Minus)):
                    if A_MD:
                        kwargs["a_minus"] = A_Minus[c]
                    else:
                        kwargs["a_minus"] = A_Minus
                    flad_m = True

                flad_p = False
                if A_Plus != None and ((isinstance(A_Plus, float)) or (c in A_Plus)):
                    if A_PD:
                        kwargs["a_plus"] = A_Plus[c]
                    else:
                        kwargs["a_plus"] = A_Plus
                    flad_p = True

                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )
                if flad_m:
                    kwargs.pop("a_minus")
                if flad_p:
                    kwargs.pop("a_plus")

            # # Get input to all layers.
            # current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)


import glfw
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
import warnings
import numpy as np

# CUDA<->GL interop is only used when the model runs on the GPU. On a CPU model
# (or a machine without CUDA) these are never touched, so import them optionally:
# the renderer falls back to plain host->GL uploads (glBufferSubData / set_data).
try:
    from cuda.bindings import driver
    from cuda.bindings import runtime
    import cupy as cp
    _CUDA_INTEROP_AVAILABLE = True
except Exception:  # cupy / cuda.bindings absent (e.g. CPU-only install)
    driver = runtime = cp = None
    _CUDA_INTEROP_AVAILABLE = False

pytorch_cp_type_map = {
    torch.float32: cp.float32,
    torch.float64: cp.float64,
    torch.int32: cp.int32,
    torch.int64: cp.int64,
    torch.uint8: cp.uint8,
    torch.bool: cp.bool,
} if _CUDA_INTEROP_AVAILABLE else {}
pytorch_opengl_type_map = {
    torch.float32: gl.GL_FLOAT,
    torch.float64: gl.GL_DOUBLE,
    torch.int32: gl.GL_INT,
    torch.uint8: gl.GL_UNSIGNED_BYTE,
    torch.bool: gl.GL_UNSIGNED_BYTE,
}
class GUINetwork(Network):
    # language=rst
    """
    Subclass of ``Network`` with added functionality for live plotting using VisPy.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opengl_vbos = {'connections': {}, 'layers': {}}
        # name -> dict describing a CUDA-registered GL buffer holding that layer's
        # FULL spike history, (T, batch, n) one byte/spike. Populated lazily by a
        # raster widget via enable_spike_history(); see step() for the write path.
        self._spike_history = {}
        # Same idea for voltages: (T, batch, n) float32, written IN PLACE by the node
        # (LIFNodes.forward updates self.v in place), so the voltage a layer computes
        # lands straight in the GL buffer with no BindsNET->viz copy. Populated by a
        # voltage widget via enable_voltage_history(); see step() for the write path.
        self._voltage_history = {}
        self._step_t = 0  # internal timestep counter (used if step() called without t)
        # True  -> model lives on the GPU: history is recorded zero-copy via CUDA<->GL
        #          interop (the node writes straight into mapped GL buffers).
        # False -> model lives on the CPU (or no CUDA): history is recorded by uploading
        #          each step's row from the host tensor into the GL buffer.
        # Resolved from the layers' device on first use (see _resolve_backend).
        self._use_cuda = None

    def _resolve_backend(self) -> bool:
        # language=rst
        """
        Decide once whether the GPU (CUDA<->GL interop) or CPU (host->GL upload) render
        path is used, based on where the layers' state tensors live. Cached so every
        per-step branch is a cheap attribute read.
        """
        if self._use_cuda is None:
            on_cuda = any(
                getattr(layer, 's', None) is not None and layer.s.is_cuda
                for layer in self.layers.values()
            )
            self._use_cuda = bool(_CUDA_INTEROP_AVAILABLE and on_cuda)
        return self._use_cuda

    def migrate(self) -> None:
        ### Migrate all layers and connections to shared buffers ###
        if not self._resolve_backend():
            # CPU model: no CUDA<->GL shared buffers. Layer state stays in host tensors
            # and is uploaded into the history GL buffers per step (see step()). The
            # per-layer s/v shared buffers are a CUDA-only mechanism unused by the
            # current renderer anyway, so there is nothing to migrate here.
            return
        for name in self.layers:
            self.migrate_layer(name)

    def migrate_layer(self, name: str) -> None:
        ### Determine which data needs a shared buffer ###
        layer = self.layers[name]
        layer_data = {}
        if isinstance(layer, Input):
            layer_data['s'] = layer.s
        elif isinstance(layer, LIFNodes):
            layer_data['s'] = layer.s
            layer_data['v'] = layer.v
        else:
            raise NotImplementedError("GUINetwork only supports Input and LIFNodes layers for now.")

        ### Create shared buffers ###
        self.opengl_vbos['layers'][name] = {}
        for data_name, data in layer_data.items():
            shared_buffer, vbo = self._create_shared_buffer(data)           # Generate buffer
            layer.__setattr__(data_name, shared_buffer)                     # Replace original tensor with shared buffer
            self.opengl_vbos['layers'][name][data_name] = vbo
            # self.opengl_vaos['layers'][name][data_name] = vao            # Map VBO to layer attribute
            # self.opengl_vao_dtypes[vao] = pytorch_opengl_type_map[data.dtype]    # Store OpenGL type for this buffer

    def _create_shared_buffer(self, org_tensor: torch.Tensor) -> tuple[Tensor, int]:
        # language=rst
        """
        Create a shared buffer for a class variable tensor/buffer.

        :param org_tensor: PyTorch tensor to create a shared buffer for.
        :return:
            ``shared_buffer``: New PyTorch tensor that shares memory with an OpenGL buffer registered with CUDA
            ``vao``: OpenGL buffer object ID that is shared with the new PyTorch tensor.
        """

        N = org_tensor.numel()
        buffer_size = N * org_tensor.element_size()

        ### Setup OpenGL buffer ###
        vbo = gl.glGenBuffers(1)  # Vertex Buffer Object
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)  # Bind to GL_ARRAY_BUFFER
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,  # Allocate buffer space
                        size=buffer_size,  # Size in bytes
                        data=None,  # No initial data
                        usage=gl.GL_DYNAMIC_DRAW)  # Frequent updates expected
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind buffer
        if gl.glIsBuffer(vbo) == 0:
            raise RuntimeError("Failed to create OpenGL buffer")

        ### Register OpenGL buffer with CUDA ###
        err, = driver.cuInit(0)  # Initialize CUDA driver
        if err != 0: raise RuntimeError(f"Failed to initialize CUDA: error code {err}")

        err, device = driver.cuDeviceGet(0)  # Get CUDA device
        if err != 0: raise RuntimeError(f"Failed to get CUDA device: error code {err}")

        err, context = driver.cuCtxCreate(None, 0, device)  # Create CUDA context
        if err != 0: raise RuntimeError(f"Failed to create CUDA context: error code {err}")

        err, cuda_resource = driver.cuGraphicsGLRegisterBuffer(
            buffer=vbo,
            Flags=2  # cuda.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        )
        if err != 0: raise RuntimeError(f"Failed to register OpenGL buffer with CUDA: error code {err}")

        err, = driver.cuGraphicsMapResources(1, cuda_resource, 0)
        if err != 0: raise RuntimeError(f"Failed to map CUDA graphics resource: error code {err}")

        err, cuda, size = driver.cuGraphicsResourceGetMappedPointer(cuda_resource)
        if err != 0: raise RuntimeError(f"Failed to get mapped pointer for CUDA graphics resource: error code {err}")

        ### Define VAO ###
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 1, pytorch_opengl_type_map[org_tensor.dtype], False, 0, None)
        gl.glBindVertexArray(0)

        ### Create PyTorch tensor from CUDA pointer ###
        cp_ptr = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(cuda), size, cuda_resource), 0)
        dtype = pytorch_cp_type_map[org_tensor.dtype]
        cp_array = cp.ndarray(N, dtype=dtype, memptr=cp_ptr)
        torch_tensor = torch.as_tensor(cp_array)                # Create tensor with shared memory location
        torch_tensor = torch_tensor.reshape(org_tensor.shape)   # Reshape to original tensor shape
        torch_tensor.copy_(org_tensor)  # Copy original tensor values to shared buffer

        return torch_tensor, vao

    def enable_spike_history(self, layer_name: str, total_timesteps: int) -> dict:
        # language=rst
        """
        Allocate a CUDA-registered GL buffer holding a layer's FULL spike history
        and route the layer's ``s`` into it so spikes are written in place (true
        zero copy -- the node's ``torch.ge(..., out=self.s)`` lands straight in the
        buffer). A full-history raster visual then reads it via ``texelFetch``.

        Layout is time-major ``(T, batch, n)``, one byte per spike. ``T`` is
        clamped so ``batch * n * T`` fits ``GL_MAX_TEXTURE_BUFFER_SIZE``.

        :param layer_name: Name of the layer to record (must already be added).
        :param total_timesteps: Desired history length (typically the full run).
        :return: ``{'vbo', 'T', 'n', 'row'}`` for the owning widget/visual.
        """
        layer = self.layers[layer_name]
        n = int(layer.n)
        batch = int(self.batch_size)
        row = batch * n                       # bytes (=elements) per timestep
        T = int(total_timesteps)

        # Cap to the driver's texture-buffer limit so glTexBuffer can address it all.
        max_texels = int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_BUFFER_SIZE))
        if row * T > max_texels:
            T = max(1, max_texels // row)
            warnings.warn(
                f"Spike history for '{layer_name}' capped to T={T} timesteps "
                f"({row}*{total_timesteps} bytes exceeds GL_MAX_TEXTURE_BUFFER_SIZE="
                f"{max_texels}). History before the cap will not be retained."
            )

        nbytes = row * T  # 1 byte per element (bool spikes)

        ### Allocate a GL buffer, zero-initialised so unwritten rows read 0 ###
        vbo = int(gl.glGenBuffers(1))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nbytes,
                        np.zeros(nbytes, dtype=np.uint8), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        if gl.glIsBuffer(vbo) == 0:
            raise RuntimeError("Failed to create spike-history GL buffer")

        if not self._resolve_backend():
            # CPU model: no CUDA registration. The layer keeps its own `s`; step()
            # uploads each timestep's spikes into this buffer with glBufferSubData.
            self._spike_history[layer_name] = {
                'vbo': vbo, 'T': T, 'n': n, 'batch': batch, 'row': row,
                'shape': tuple(layer.shape),
            }
            return {'vbo': vbo, 'T': T, 'n': n, 'row': row}

        ### Register with CUDA, reusing PyTorch's existing context (NONE flag: keep
        ### prior contents -- this is an accumulating history, not WRITE_DISCARD) ###
        self._ensure_cuda_context()
        err, res = driver.cuGraphicsGLRegisterBuffer(buffer=vbo, Flags=0)
        if err != 0:
            raise RuntimeError(f"cuGraphicsGLRegisterBuffer (history) failed: {err}")

        self._spike_history[layer_name] = {
            'vbo': vbo, 'res': res, 'T': T, 'n': n, 'batch': batch, 'row': row,
            'shape': tuple(layer.shape),  # per-sample shape, e.g. (n,)
            # Scratch s used once t exceeds the (possibly capped) capacity, so the
            # sim keeps running -- it just stops recording past T.
            'scratch': torch.zeros(batch, *layer.shape, dtype=torch.bool,
                                   device=layer.s.device),
        }
        return {'vbo': vbo, 'T': T, 'n': n, 'row': row}

    def _ensure_cuda_context(self) -> None:
        # Reuse the current (PyTorch/cupy-created) CUDA context instead of calling
        # cuCtxCreate per buffer (the bug in _create_shared_buffer). A CUDA model is
        # already resident, so a context exists; touch cupy if somehow it doesn't.
        (err,) = driver.cuInit(0)
        if err != 0:
            raise RuntimeError(f"cuInit failed: {err}")
        err, ctx = driver.cuCtxGetCurrent()
        if err != 0 or int(ctx) == 0:
            cp.zeros(1)  # force a context onto this thread
            err, ctx = driver.cuCtxGetCurrent()
            if err != 0 or int(ctx) == 0:
                raise RuntimeError("No current CUDA context for GL interop")

    def _map_history(self, h: dict) -> torch.Tensor:
        # Map the GL buffer (CUDA takes ownership so the node can write it) and wrap
        # it as a (T, batch, *shape) torch view. The mapped pointer MAY change
        # between maps, but in practice is stable, so cache the wrapped view and
        # only rebuild it when the pointer actually moves (torch.as_tensor over the
        # CUDA-array-interface is not free per step).
        (err,) = driver.cuGraphicsMapResources(1, h['res'], 0)
        if err != 0:
            raise RuntimeError(f"map spike history failed: {err}")
        err, ptr, size = driver.cuGraphicsResourceGetMappedPointer(h['res'])
        if err != 0:
            raise RuntimeError(f"get mapped pointer (history) failed: {err}")
        if h.get('ptr') == int(ptr) and h.get('view') is not None:
            return h['view']
        n_elems = h['row'] * h['T']
        cp_ptr = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(int(ptr), size, h['res']), 0)
        cp_arr = cp.ndarray(n_elems, dtype=cp.bool_, memptr=cp_ptr)
        view = torch.as_tensor(cp_arr).view(h['T'], h['batch'], *h['shape'])
        h['ptr'] = int(ptr)
        h['view'] = view
        return view

    def enable_voltage_history(self, layer_name: str, total_timesteps: int) -> dict:
        # language=rst
        """
        Allocate a CUDA-registered GL buffer holding a layer's FULL voltage history
        and arrange for the node to write each timestep's voltage straight into it.

        Unlike spikes (written via ``torch.ge(..., out=self.s)``), voltage is a
        recurrent state: ``v[t]`` is computed from ``v[t-1]``. So :meth:`step` seeds
        row ``t`` with row ``t-1`` (a buffer-internal device copy) and points
        ``layer.v`` at that row; :class:`LIFNodes` then updates ``v`` *in place*
        (see ``nodes.py``), so the voltage it computes lands directly in the GL
        buffer -- no copy of the value out of BindsNET into a viz object. A
        full-history voltage visual reads it back via ``texelFetch``.

        Layout is time-major ``(T, batch, n)`` float32. ``T`` is clamped so
        ``batch * n * T`` fits ``GL_MAX_TEXTURE_BUFFER_SIZE``.

        :param layer_name: Name of the layer to record (must already be added).
        :param total_timesteps: Desired history length (typically the full run).
        :return: ``{'vbo', 'T', 'n', 'row'}`` for the owning widget/visual.
        """
        layer = self.layers[layer_name]
        n = int(layer.n)
        batch = int(self.batch_size)
        row = batch * n                       # floats (=texels) per timestep
        T = int(total_timesteps)

        # Cap to the driver's texture-buffer limit (in texels; one float == one R32F
        # texel) so glTexBuffer can address it all.
        max_texels = int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_BUFFER_SIZE))
        if row * T > max_texels:
            T = max(1, max_texels // row)
            warnings.warn(
                f"Voltage history for '{layer_name}' capped to T={T} timesteps "
                f"({row}*{total_timesteps} floats exceeds GL_MAX_TEXTURE_BUFFER_SIZE="
                f"{max_texels}). History before the cap will not be retained."
            )

        nbytes = row * T * 4  # float32

        ### Allocate a GL buffer, zero-initialised so unwritten rows read 0 ###
        vbo = int(gl.glGenBuffers(1))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nbytes,
                        np.zeros(nbytes, dtype=np.uint8), gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        if gl.glIsBuffer(vbo) == 0:
            raise RuntimeError("Failed to create voltage-history GL buffer")

        # Running min/max of the recorded voltage, updated each step from the freshly
        # written row. The voltage widget reads these (via .item()) to size a dynamic
        # y-axis. Lives on the layer's device -- on the GPU they stay GPU scalars (the
        # only host sync is the widget's .item()); on the CPU they're host scalars.
        vmin = torch.full((), float('inf'), dtype=torch.float32, device=layer.v.device)
        vmax = torch.full((), float('-inf'), dtype=torch.float32, device=layer.v.device)

        if not self._resolve_backend():
            # CPU model: no CUDA registration. The layer keeps its own (recurrent) `v`;
            # step() uploads each timestep's voltage into this buffer and folds it into
            # vmin/vmax on the host.
            self._voltage_history[layer_name] = {
                'vbo': vbo, 'T': T, 'n': n, 'batch': batch, 'row': row,
                'shape': tuple(layer.shape), 'vmin': vmin, 'vmax': vmax,
            }
            return {'vbo': vbo, 'T': T, 'n': n, 'row': row, 'vmin': vmin, 'vmax': vmax}

        ### Register with CUDA (NONE flag: keep prior contents -- this is an
        ### accumulating history, and rows carry voltage forward, not WRITE_DISCARD) ###
        self._ensure_cuda_context()
        err, res = driver.cuGraphicsGLRegisterBuffer(buffer=vbo, Flags=0)
        if err != 0:
            raise RuntimeError(f"cuGraphicsGLRegisterBuffer (voltage) failed: {err}")

        self._voltage_history[layer_name] = {
            'vbo': vbo, 'res': res, 'T': T, 'n': n, 'batch': batch, 'row': row,
            'shape': tuple(layer.shape),  # per-sample shape, e.g. (n,)
            'vmin': vmin, 'vmax': vmax,   # observed voltage range (in-place updated)
            # Scratch v used once t exceeds the (possibly capped) capacity, so the
            # sim's voltage recurrence keeps running -- it just stops recording.
            'scratch': torch.zeros(batch, *layer.shape, dtype=torch.float32,
                                   device=layer.v.device),
        }
        return {'vbo': vbo, 'T': T, 'n': n, 'row': row, 'vmin': vmin, 'vmax': vmax}

    def _map_voltage_history(self, h: dict) -> torch.Tensor:
        # Map the GL buffer (CUDA takes ownership so the node can write it) and wrap
        # it as a (T, batch, *shape) float32 torch view. Caches the wrapped view and
        # rebuilds only when the mapped pointer actually moves (see _map_history).
        (err,) = driver.cuGraphicsMapResources(1, h['res'], 0)
        if err != 0:
            raise RuntimeError(f"map voltage history failed: {err}")
        err, ptr, size = driver.cuGraphicsResourceGetMappedPointer(h['res'])
        if err != 0:
            raise RuntimeError(f"get mapped pointer (voltage) failed: {err}")
        if h.get('ptr') == int(ptr) and h.get('view') is not None:
            return h['view']
        n_elems = h['row'] * h['T']
        cp_ptr = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(int(ptr), size, h['res']), 0)
        cp_arr = cp.ndarray(n_elems, dtype=cp.float32, memptr=cp_ptr)
        view = torch.as_tensor(cp_arr).view(h['T'], h['batch'], *h['shape'])
        h['ptr'] = int(ptr)
        h['view'] = view
        return view

    def reset_history(self) -> None:
        # language=rst
        """
        Zero the spike/voltage history GL buffers and the running voltage min/max,
        so a reset clears every recorded sample (not just the live state). Maps each
        buffer with CUDA, zeros it in place via the cached torch view, and hands it
        back to GL. Safe to call between steps (the timer loop is single-threaded, so
        no step is concurrently holding a map).
        """
        if not self._resolve_backend():
            # CPU model: zero each GL buffer with a host upload, reset the running
            # voltage range, and rewind the step counter.
            for h in self._spike_history.values():
                zeros = np.zeros(h['row'] * h['T'], dtype=np.uint8)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, h['vbo'])
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, zeros.nbytes, zeros)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            for h in self._voltage_history.values():
                zeros = np.zeros(h['row'] * h['T'], dtype=np.float32)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, h['vbo'])
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, zeros.nbytes, zeros)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
                h['vmin'].fill_(float('inf'))
                h['vmax'].fill_(float('-inf'))
            self._step_t = 0
            return
        for h in self._spike_history.values():
            view = self._map_history(h)
            view.zero_()
            (err,) = driver.cuGraphicsUnmapResources(1, h['res'], 0)
            if err != 0:
                raise RuntimeError(f"unmap spike history (reset) failed: {err}")
        for h in self._voltage_history.values():
            view = self._map_voltage_history(h)
            view.zero_()
            (err,) = driver.cuGraphicsUnmapResources(1, h['res'], 0)
            if err != 0:
                raise RuntimeError(f"unmap voltage history (reset) failed: {err}")
            h['vmin'].fill_(float('inf'))
            h['vmax'].fill_(float('-inf'))
        self._step_t = 0

    def _upload_spike_row(self, layer_name: str, t: int) -> None:
        # CPU path: copy this timestep's spikes from the layer's host tensor into row
        # `t` of the spike-history GL buffer (R8UI, one byte/spike). Past the capacity
        # cap there is no row to write, so recording simply stops.
        h = self._spike_history[layer_name]
        if t >= h['T']:
            return
        row = self.layers[layer_name].s.detach().to(torch.uint8).contiguous() \
            .view(-1).cpu().numpy()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, h['vbo'])
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, t * h['row'], row.nbytes, row)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _upload_voltage_row(self, layer_name: str, t: int) -> None:
        # CPU path: copy this timestep's voltage into row `t` of the voltage-history
        # GL buffer (R32F) and fold it into the running min/max (all on the host). The
        # layer keeps its own recurrent `v`, so v[t] is already computed from v[t-1].
        h = self._voltage_history[layer_name]
        v = self.layers[layer_name].v
        torch.minimum(h['vmin'], v.min(), out=h['vmin'])
        torch.maximum(h['vmax'], v.max(), out=h['vmax'])
        if t >= h['T']:
            return
        row = v.detach().to(torch.float32).contiguous().view(-1).cpu().numpy()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, h['vbo'])
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, t * h['row'] * 4, row.nbytes, row)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _step_cpu(self, input: Dict[str, torch.Tensor], t: int) -> None:
        # CPU render path: a plain simulation step (the layers keep their own `s`/`v`,
        # whose values persist across steps -- so _get_inputs() reads last step's
        # spikes and LIFNodes carries voltage forward, no buffer repointing needed),
        # followed by a host->GL upload of whatever is being recorded.
        current_inputs = {}
        current_inputs.update(self._get_inputs())
        for l in self.layers:
            if l in input:
                if l in current_inputs:
                    current_inputs[l] += input[l]
                else:
                    current_inputs[l] = input[l]

            if l in current_inputs:
                self.layers[l].forward(x=current_inputs[l])
            else:
                self.layers[l].forward(
                    x=torch.zeros(
                        self.layers[l].s.shape, device=self.layers[l].s.device
                    )
                )

            # Record this timestep into the (host-backed) GL history buffers.
            if l in self._spike_history:
                self._upload_spike_row(l, t)
            if l in self._voltage_history:
                self._upload_voltage_row(l, t)

        for c in self.connections:
            self.connections[c].update(reward=1, learning=True)        # TODO: TEMPORARY arguments

        self._step_t = t + 1

    def step(self, input: Dict[str, torch.Tensor], t: int = None) -> None:
        ### Simulate network activity for one time step ###
        if t is None:
            t = self._step_t

        if not self._resolve_backend():
            return self._step_cpu(input, t)

        # Map any spike-history buffers and point each layer's `s` at the PREVIOUS
        # timestep's row, so _get_inputs() / connections read last step's spikes
        # through a valid (currently-mapped) pointer. At t==0 this indexes the last
        # (still-zero) row -- correct: no spikes precede the run.
        views = {}
        for name, h in self._spike_history.items():
            view = self._map_history(h)
            views[name] = view
            # Previous timestep's spikes (valid, mapped) for _get_inputs. Past the
            # capacity cap, fall back to scratch (no recorded history to read).
            if 0 <= t - 1 < h['T']:
                self.layers[name].s = view[t - 1]
            else:
                self.layers[name].s = h['scratch']

        # Map any voltage-history buffers so the layer's in-place voltage update can
        # land in the GL buffer (the actual repoint happens just before forward()).
        vviews = {}
        for name, h in self._voltage_history.items():
            vviews[name] = self._map_voltage_history(h)

        current_inputs = {}
        current_inputs.update(self._get_inputs())
        for l in self.layers:
            # Update each layer of nodes.
            if l in input:
                if l in current_inputs:
                    current_inputs[l] += input[l]
                else:
                    current_inputs[l] = input[l]

            # Point a recorded layer's `s` at THIS timestep's row so the in-place
            # spike write (torch.ge(out=self.s)) accumulates straight into history.
            # Past the capacity cap, write to scratch instead (recording stopped).
            if l in views:
                h = self._spike_history[l]
                self.layers[l].s = views[l][t] if t < h['T'] else h['scratch']

            # Point a recorded layer's `v` at THIS timestep's row, seeded with the
            # PREVIOUS timestep's voltage (recurrent state carried forward). The node
            # then updates `v` in place, so the computed voltage lands in the GL
            # buffer with no copy out of BindsNET. Past the capacity cap, fall back to
            # scratch (recording stopped) but keep the recurrence alive.
            if l in vviews:
                h = self._voltage_history[l]
                if t < h['T']:
                    dst = vviews[l][t]
                    dst.copy_(self.layers[l].v if t == 0 else vviews[l][t - 1])
                    self.layers[l].v = dst
                else:
                    if t == h['T']:
                        h['scratch'].copy_(vviews[l][h['T'] - 1])
                    self.layers[l].v = h['scratch']

            if l in current_inputs:
                self.layers[l].forward(x=current_inputs[l])
            else:
                self.layers[l].forward(
                    x=torch.zeros(
                        self.layers[l].s.shape, device=self.layers[l].s.device
                    )
                )

            # Fold this step's voltage into the recorded layer's running min/max
            # while the row is still mapped (forward() just wrote it in place). GPU
            # reductions only -- no host sync here; the widget syncs when it draws.
            if l in vviews:
                h = self._voltage_history[l]
                if t < h['T']:
                    row = vviews[l][t]
                    torch.minimum(h['vmin'], row.min(), out=h['vmin'])
                    torch.maximum(h['vmax'], row.max(), out=h['vmax'])

        for c in self.connections:
            self.connections[c].update(reward=1, learning=True)        # TODO: TEMPORARY arguments

        # Hand the buffers back to OpenGL so the raster visual can draw them.
        for name, h in self._spike_history.items():
            (err,) = driver.cuGraphicsUnmapResources(1, h['res'], 0)
            if err != 0:
                raise RuntimeError(f"unmap spike history failed: {err}")

        # Same for voltage-history buffers (written in place by the node above).
        for name, h in self._voltage_history.items():
            (err,) = driver.cuGraphicsUnmapResources(1, h['res'], 0)
            if err != 0:
                raise RuntimeError(f"unmap voltage history failed: {err}")

        self._step_t = t + 1

    def run(self, inputs: Dict[str, torch.Tensor], time: int, **kwargs) -> None:
        raise NotImplementedError(
            "GUI Network does not currently support the 'run' method."
            "Please use the 'step' function to run the network one time step at a time"
        )
