# TODO(dangnh36): Multiple profiles implementation
# TODO(dangnh36): Different threads use same engine but different execution context (streaming).
# Current implementation: different threads use different engine.
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

# trt.init_libnvinfer_plugins(None, "")
# import pycuda.autoinit

TRT_LOGGER = trt.Logger()
logger = logging.getLogger(__name__)


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
    ):
        self.host = host_mem
        self.device = device_mem
        self.binding_idx = binding_idx
        self.shape = shape
        self.volume = volume
        self.size = size

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTEngine:

    def __init__(self, weight_path, gpu_id=0):
        print(f"Loading TensorRT engine on GPU {gpu_id} from {weight_path}")
        self.device_ctx = cuda.Device(gpu_id).make_context()
        self.stream = cuda.Stream()
        self.engine = self._load_engine(weight_path)
        logger.info(
            "HAS IMPLICIT BATCH DIMENSION ? %s",
            self.engine.has_implicit_batch_dimension,
        )
        self.context = self.engine.create_execution_context()

        num_profiles = self.engine.num_optimization_profiles
        first_tensor_name = self.engine.get_tensor_name(0)
        for i in range(num_profiles):
            _min, _opt, _max = self.engine.get_tensor_profile_shape(
                first_tensor_name, profile_index=i
            )
            if i == self.context.active_optimization_profile:
                engine_max_batch_size = _max[0]
                logger.info("*** MAIN PROFILE ***")
                logger.info(
                    "Profile %s: Min %s Opt %s Max %s\n----------", i, _min, _opt, _max
                )

        # Max batch size
        # Using this batch size gives maximum performance
        self.max_batch_size = engine_max_batch_size

        logger.info("ENGINE MAX BATCH SIZE: %d", self.max_batch_size)

        input_mems, output_mems, self.bindings, input_shapes, output_shapes = (
            self._allocate_buffers()
        )
        self.input_mems: List[HostDeviceMem] = input_mems
        self.output_mems: List[HostDeviceMem] = output_mems

        # set default binding shape with maximum batch size
        for i, input_mem in enumerate(self.input_mems):
            self.context.set_binding_shape(
                input_mem.binding_idx, (self.max_batch_size, *input_mem.shape)
            )

    def _load_engine(self, trt_file):
        """Load tensorrt engine from file."""
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self):
        """Allocate device memory space for data."""
        inputs = []
        outputs = []
        bindings = []
        input_shapes = []
        output_shapes = []
        logger.info("*** BINDING LIST ***")
        for binding_idx, binding in enumerate(self.engine):
            tensor_name = self.engine.get_tensor_name(binding_idx)
            tensor_shape = self.engine.get_tensor_shape(binding)
            shape = tensor_shape[1:]
            volume = trt.volume(shape)
            size = volume * self.max_batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            logger.warning(
                "binding: %s name=%s shape=%s dtype=%s",
                binding,
                tensor_name,
                tensor_shape,
                dtype,
            )

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(
                    HostDeviceMem(
                        host_mem, device_mem, binding_idx, shape, volume, size
                    )
                )
                input_shapes.append(tensor_shape)
            else:
                outputs.append(
                    HostDeviceMem(
                        host_mem, device_mem, binding_idx, shape, volume, size
                    )
                )
                output_shapes.append(tensor_shape)
        return inputs, outputs, bindings, input_shapes, output_shapes

    def __del__(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.device_ctx.pop()
        # self.device_ctx.detach()

    def forward(self, inputs):
        batch_size = inputs[0].shape[0]
        assert len(inputs) == len(self.input_mems)

        # copy data to Host buffer
        for i, input_mem in enumerate(self.input_mems):
            np.copyto(input_mem.host[: inputs[i].size], inputs[i], casting="no")

        # Make self the active context, pushing it on top of the context stack.
        self.device_ctx.push()

        # copy input data to the GPU.
        for i, input_mem in enumerate(self.input_mems):
            cuda.memcpy_htod_async(
                input_mem.device, input_mem.host[: inputs[i].size], self.stream
            )
            self.context.set_binding_shape(
                input_mem.binding_idx, (batch_size, *input_mem.shape)
            )

        # Run inference.
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Transfer predictions back from the GPU to Host
        for i, output_mem in enumerate(self.output_mems):
            cuda.memcpy_dtoh_async(
                output_mem.host[: batch_size * output_mem.volume],
                output_mem.device,
                self.stream,
            )

        # Synchronize the stream
        self.stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.device_ctx.pop()

        # parse outputs
        outputs = [
            np.reshape(
                out_mem.host[: batch_size * out_mem.volume],
                (-1, *out_mem.shape),
            )
            for out_mem in self.output_mems
        ]
        return outputs


######## TEST ########

_TEST_WEIGHT_PATH = "assets/EXP23_X3DM_ALLGTV3_ep=2_step=36000.engine"
_TEST_GPU_ID = 0
_TEST_SHAPE = (1, 224, 448, 448)
_TEST_CONCURRENT_LEVEL = "process"
_TEST_NUM_INSTANCES = 2
_TEST_DURATION = 30
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
    engine = TRTEngine(weight_path=_TEST_WEIGHT_PATH, gpu_id=_TEST_GPU_ID)

    # prepair input
    inp = (np.random.rand(batch_size, *_TEST_SHAPE) * 255).astype(np.uint8)

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
