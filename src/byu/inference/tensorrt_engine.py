"""
TENSORRT 10.X ENGINE
EXPERIMENTAL, FAST PROTOTYPING, NOT CAREFULLY TESTED, CONTAINS MANY BUGS (SURE, I FACE SOME OF THEME)
DEVELOPED WITH HELPS OF CLAUDE + CHATGPT :)
NOT RECOMMEND TO WASTE YOUR TIME ON THIS...
"""

import logging
import threading
import time
from typing import Any, List, Optional

import pycuda.driver as cuda
import tensorrt as trt
import torch

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Thread-local storage for CUDA context management
_thread_local = threading.local()


class HostDeviceMem(object):
    """Wrapper for host and device memory for TensorRT related code."""

    def __init__(
        self,
        host_mem: Any,
        device_mem: Any,
        binding_idx: Optional[int] = None,
        shape=None,
        volume=None,
        size=None,
        name: Optional[str] = None,
    ):
        self.host = host_mem
        self.device = device_mem
        self.binding_idx = binding_idx
        self.shape = shape
        self.volume = volume
        self.size = size
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class ThreadsafeTRTEngine:

    def __init__(self, weight_path, gpu_id=0):
        self.gpu_id = gpu_id
        self.weight_path = weight_path

        # Thread-safe initialization
        self._init_lock = threading.Lock()

        print(f"Loading TensorRT engine on GPU {gpu_id} from {weight_path}")

        # Initialize CUDA context safely
        self._ensure_cuda_context()

        # Load engine
        self.engine = self._load_engine(weight_path)
        logger.info(
            "HAS IMPLICIT BATCH DIMENSION ? %s",
            self.engine.has_implicit_batch_dimension,
        )

        # Create execution context - each engine needs its own
        self.context = self.engine.create_execution_context()

        # Get optimization profiles info
        num_profiles = self.engine.num_optimization_profiles

        # Get input tensor names (TensorRT 10+ way)
        input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]

        if input_names:
            first_tensor_name = input_names[0]
            for i in range(num_profiles):
                _min, _opt, _max = self.engine.get_tensor_profile_shape(
                    first_tensor_name, profile_index=i
                )
                if i == self.context.active_optimization_profile:
                    engine_max_batch_size = _max[0]
                    logger.info("*** MAIN PROFILE ***")
                    logger.info(
                        "Profile %s: Min %s Opt %s Max %s\n----------",
                        i,
                        _min,
                        _opt,
                        _max,
                    )

            # Max batch size
            self.max_batch_size = engine_max_batch_size
        else:
            self.max_batch_size = 1

        logger.info("ENGINE MAX BATCH SIZE: %d", self.max_batch_size)

        # Allocate buffers
        input_mems, output_mems, self.tensor_addresses, input_shapes, output_shapes = (
            self._allocate_buffers()
        )
        self.input_mems: List[HostDeviceMem] = input_mems
        self.output_mems: List[HostDeviceMem] = output_mems

        # Set default tensor shapes with maximum batch size
        for input_mem in self.input_mems:
            if input_mem.name:
                self.context.set_input_shape(
                    input_mem.name, (self.max_batch_size, *input_mem.shape)
                )

    def _ensure_cuda_context(self):
        """Ensure CUDA context is properly initialized for this thread."""
        with self._init_lock:
            if not hasattr(_thread_local, "cuda_contexts"):
                _thread_local.cuda_contexts = {}

            if self.gpu_id not in _thread_local.cuda_contexts:
                # Only initialize pycuda.autoinit once per thread
                if not hasattr(_thread_local, "pycuda_initialized"):
                    import pycuda.autoinit

                    _thread_local.pycuda_initialized = True

                # Create device context for this GPU
                device = cuda.Device(self.gpu_id)
                context = device.make_context()
                _thread_local.cuda_contexts[self.gpu_id] = context

                # Create stream for this context
                self.stream = cuda.Stream()
            else:
                # Use existing context
                _thread_local.cuda_contexts[self.gpu_id].push()
                self.stream = cuda.Stream()
                _thread_local.cuda_contexts[self.gpu_id].pop()

    def _get_cuda_context(self):
        """Get the CUDA context for this GPU."""
        return _thread_local.cuda_contexts.get(self.gpu_id)

    def _load_engine(self, trt_file):
        """Load tensorrt engine from file."""
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self):
        """Allocate device memory space for data."""
        inputs = []
        outputs = []
        tensor_addresses = {}
        input_shapes = []
        output_shapes = []

        logger.info("*** TENSOR LIST ***")

        # Ensure we're in the right context for memory allocation
        cuda_context = self._get_cuda_context()
        if cuda_context:
            cuda_context.push()

        try:
            # Iterate through all tensors using the new API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape = self.engine.get_tensor_shape(tensor_name)
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

                # Handle dynamic shapes - use max batch size for allocation
                if tensor_shape[0] == -1:  # Dynamic batch dimension
                    raise Exception
                    shape = tensor_shape[1:]
                    volume = trt.volume(shape)
                    size = volume * self.max_batch_size
                else:
                    shape = tensor_shape[1:] if len(tensor_shape) > 1 else tensor_shape
                    volume = trt.volume(tensor_shape)
                    size = volume

                logger.warning(
                    "tensor: %s shape=%s dtype=%s mode=%s",
                    tensor_name,
                    tensor_shape,
                    dtype,
                    tensor_mode,
                )

                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                # Store tensor address mapping
                tensor_addresses[tensor_name] = device_mem

                # Create HostDeviceMem object
                host_device_mem = HostDeviceMem(
                    host_mem, device_mem, i, shape, volume, size, tensor_name
                )

                # Append to the appropriate list
                if tensor_mode == trt.TensorIOMode.INPUT:
                    inputs.append(host_device_mem)
                    input_shapes.append(tensor_shape)
                else:
                    outputs.append(host_device_mem)
                    output_shapes.append(tensor_shape)

        finally:
            if cuda_context:
                cuda_context.pop()

        return inputs, outputs, tensor_addresses, input_shapes, output_shapes

    def __del__(self):
        """Clean up resources."""
        # Free device memory
        try:
            cuda_context = self._get_cuda_context()
            if cuda_context:
                cuda_context.push()
                # Free allocated memory
                for mem in getattr(self, "input_mems", []):
                    if hasattr(mem, "device") and mem.device:
                        mem.device.free()
                for mem in getattr(self, "output_mems", []):
                    if hasattr(mem, "device") and mem.device:
                        mem.device.free()
                cuda_context.pop()
        except:
            pass  # Ignore cleanup errors

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs):
        """
        Forward pass with PyTorch GPU tensors.

        Args:
            inputs: List of PyTorch tensors on GPU (CUDA device)

        Returns:
            List of PyTorch tensors on GPU
        """
        # Validate inputs
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        for i, inp in enumerate(inputs):
            if not isinstance(inp, torch.Tensor):
                raise TypeError(f"Input {i} must be a PyTorch tensor, got {type(inp)}")
            if not inp.is_cuda:
                raise ValueError(
                    f"Input {i} must be on CUDA device, got device: {inp.device}"
                )

        batch_size = inputs[0].shape[0]
        assert len(inputs) == len(self.input_mems)

        outputs = []

        # Get the CUDA context for this engine
        cuda_context = self._get_cuda_context()
        if not cuda_context:
            raise RuntimeError("CUDA context not available")

        try:
            # Make this engine's context active
            cuda_context.push()

            # Copy inputs
            for i, input_mem in enumerate(self.input_mems):
                input_tensor = inputs[i].contiguous()
                expected_size = batch_size * input_mem.volume
                if input_tensor.numel() != expected_size:
                    raise ValueError(
                        f"Input {i} size mismatch. Expected {expected_size}, got {input_tensor.numel()}"
                    )

                input_ptr = input_tensor.data_ptr()

                # Copy from PyTorch tensor to TensorRT input buffer
                cuda.memcpy_dtod_async(
                    input_mem.device,
                    input_ptr,
                    input_tensor.numel() * input_tensor.element_size(),
                    self.stream,
                )

                if input_mem.name:
                    self.context.set_input_shape(
                        input_mem.name, (batch_size, *input_mem.shape)
                    )
                    self.context.set_tensor_address(input_mem.name, input_mem.device)

            # Set output tensor addresses
            for output_mem in self.output_mems:
                if output_mem.name:
                    self.context.set_tensor_address(output_mem.name, output_mem.device)

            # Execute inference
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT inference execution failed")

            # Wait for inference to complete before copying outputs
            self.stream.synchronize()

            # Copy outputs
            for out_mem in self.output_mems:
                if out_mem.name:
                    actual_shape = tuple(self.context.get_tensor_shape(out_mem.name))
                else:
                    actual_shape = (batch_size, *out_mem.shape)

                trt_dtype = (
                    self.engine.get_tensor_dtype(out_mem.name)
                    if out_mem.name
                    else trt.float32
                )

                torch_dtype = self._get_torch_dtype_from_trt(trt_dtype)

                output_tensor = torch.empty(
                    actual_shape, dtype=torch_dtype, device=inputs[0].device
                )
                output_ptr = output_tensor.data_ptr()
                copy_size = output_tensor.numel() * output_tensor.element_size()

                cuda.memcpy_dtod_async(
                    output_ptr,
                    out_mem.device,
                    copy_size,
                    self.stream,
                )

                outputs.append(output_tensor)

            # Final synchronization to ensure all outputs are ready
            self.stream.synchronize()

        finally:
            # Always pop the context to restore previous state
            cuda_context.pop()

        return outputs

    def _get_torch_dtype_from_trt(self, trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        dtype_mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        return dtype_mapping.get(trt_dtype, torch.float32)

    def warmup(self, input_shapes=None, num_warmup=10):
        """
        Warmup the TensorRT engine with dummy data.

        Args:
            input_shapes: List of input shapes for warmup. If None, uses max batch size.
            num_warmup: Number of warmup iterations.
        """
        if input_shapes is None:
            # Use default shapes with batch size 1
            input_shapes = [(1, *mem.shape) for mem in self.input_mems]

        # Create dummy PyTorch tensors on GPU
        dummy_inputs = []
        for i, shape in enumerate(input_shapes):
            # Determine input dtype
            trt_dtype = (
                self.engine.get_tensor_dtype(self.input_mems[i].name)
                if self.input_mems[i].name
                else trt.float32
            )
            torch_dtype = self._get_torch_dtype_from_trt(trt_dtype)

            dummy_tensor = torch.randn(
                shape, dtype=torch_dtype, device=f"cuda:{self.gpu_id}"
            )
            dummy_inputs.append(dummy_tensor)

        print(f"Warming up TensorRT engine with {num_warmup} iterations...")
        for _ in range(num_warmup):
            _ = self.forward(dummy_inputs)
        print("Warmup completed.")

    def get_max_batch_size(self):
        """Get the maximum batch size supported by this engine."""
        return self.max_batch_size

    def get_input_shapes(self):
        """Get the expected input shapes (without batch dimension)."""
        return [mem.shape for mem in self.input_mems]

    def get_output_shapes(self):
        """Get the expected output shapes (without batch dimension)."""
        return [mem.shape for mem in self.output_mems]


######## TEST ########

_TEST_WEIGHT_PATH = "assets/EXP23_X3DM_ALLGTV3_ep2_step36000.engine"
_TEST_GPU_ID = 0
_TEST_SHAPE = (1, 224, 448, 448)
_TEST_CONCURRENT_LEVEL = "process"
_TEST_NUM_INSTANCES = 1
_TEST_DURATION = 60
_TEST_BATCH_SIZE = 1


def benchmark_worker(
    start_event, stop_event, output_queue, ready_event, batch_size, worker_idx
):
    try:
        from setproctitle import setproctitle

        setproctitle(f"benchmark_worker_{worker_idx}")
    except:
        pass

    import pycuda.autoinit
    from tqdm import tqdm

    # load engine
    engine = ThreadsafeTRTEngine(weight_path=_TEST_WEIGHT_PATH, gpu_id=_TEST_GPU_ID)
    device = torch.device(f"cuda:{_TEST_GPU_ID}")
    # prepair input
    inp = (torch.rand(batch_size, *_TEST_SHAPE) * 255).to(device).float()

    # warm up
    print("Start warmup")
    for i in tqdm(range(10)):
        _output = engine.forward(inp)
    print("End warmup")
    # ensure warmup end before start real benchmark
    assert not start_event.is_set()
    time.sleep(1)
    ready_event.set()

    print("Worker waiting for start event..")
    count = 0
    start_event.wait()
    print("Worker start event indicated!")
    start = time.time()
    while not stop_event.is_set():
        output = engine.forward(inp)
        # print(output.shape)
        count += 1
    end = time.time()

    print("Benchmark worker finished!")
    output_queue.put([start, end, count])


def benchmark():
    try:
        from setproctitle import setproctitle

        setproctitle("benchmark")
    except:
        pass

    import multiprocessing as mp

    import pycuda.autoinit

    mp.set_start_method("spawn", force=True)
    import queue
    import threading

    if _TEST_CONCURRENT_LEVEL == "thread":
        start_event = threading.Event()
        stop_event = threading.Event()
        worker_cls = threading.Thread
        output_queue = queue.Queue()
    elif _TEST_CONCURRENT_LEVEL == "process":
        start_event = mp.Event()
        stop_event = mp.Event()
        worker_cls = mp.Process
        output_queue = mp.Queue()
    else:
        raise AssertionError

    workers = []
    ready_events = []
    for worker_id in range(_TEST_NUM_INSTANCES):
        ready_event = mp.Event()
        worker = worker_cls(
            group=None,
            target=benchmark_worker,
            args=(
                start_event,
                stop_event,
                output_queue,
                ready_event,
                _TEST_BATCH_SIZE,
                worker_id,
            ),
            kwargs={},
        )
        worker.start()
        workers.append(worker)
        ready_events.append(ready_event)

    print("Waiting for all workers to be in ready mode..")
    while True:
        if all([event.is_set() for event in ready_events]):
            break
    print("GLOBAL START!")
    start_event.set()

    time.sleep(_TEST_DURATION)
    stop_event.set()
    for worker in workers:
        worker.join()

    outputs = []
    while not output_queue.empty():
        outputs.append(output_queue.get())

    worker_starts = [e[0] for e in outputs]
    worker_ends = [e[1] for e in outputs]
    counts = [e[2] for e in outputs]
    global_start = min(worker_starts)
    global_end = max(worker_ends)
    global_take = global_end - global_start
    global_count = sum(counts) * _TEST_BATCH_SIZE
    avg_take = global_take / global_count

    print("Total time:", global_take, "s")
    print("Total inferences:", global_count)
    print("Avg time per inference:", round(avg_take * 1000, 2), "ms")
    print("FPS/Throughput:", 1.0 / avg_take)


if __name__ == "__main__":
    # throughput: 1089 inf/s
    benchmark()
