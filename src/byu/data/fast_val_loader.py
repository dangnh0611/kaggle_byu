import gc
import logging
import multiprocessing as mp
import os
import time
from typing import Callable, List, Tuple

import numpy as np
import torch
from yagm.utils.concurrent import ShmTensor, StoppableProcess

from byu.data.io import (
    MultithreadOpencvTomogramLoader,
    OpencvTomogramLoader,
    load_spaced_val_sample,
    load_spaced_val_sample_cache,
)

logger = logging.getLogger(__name__)


class FastValTomoLoader:
    """
    Loader to speedup dataloading with prefetch technique using multiple processes and shared memory.
    """

    def __init__(
        self,
        all_tomos_meta,
        data_dir: str,
        io_backend: str,
        target_spacing: Tuple[float, float, float],
        transform: Callable,
        prefetch: int = 3,
        keep: int = 5,
        interpolation_mode: str = "trilinear",
        shm_prefix: str = "",
        num_workers: int = 1,
    ):
        """
        Args:
            all_tomos_meta: dictionary mapping tomo_id to a dict with metadata such as ori_spacing, ori_shape, target_spacing, ori_kpts,..
            data_dir: tomogram root directory
            io_backend: cv2 or cv2_seq
            target_spacing: sampling from ori_spacing specified in tomo metadata to this target spacing, specified 3 values corresponding to (Z, Y, X)
            transform: MONAI transform applied to original tomogram, typically do Spacing + Patches Cropping
            prefetch: number of whole tomogram to prefetch, high value increase RAM usage significantly
            keep: number of tomograms to keep in shared memory
            interpolation_mode: spacing/resampling interpolation mode
            shm_prefix: prefix of shared memory name to prevent confliction in case of multiple training intances run simultaneously
            num_workers: number of prefetch workers
        """
        assert prefetch >= 0 and keep > prefetch
        self.io_backend = io_backend
        self.target_spacing = target_spacing
        self.prefetch = prefetch
        self.keep = keep
        if io_backend == "cache":
            assert transform is None
        self.interpolation_mode = interpolation_mode
        self.shm_prefix = shm_prefix
        self.num_workers = num_workers
        self.all_tomo_metas = all_tomos_meta
        self.all_tomo_ids = list(all_tomos_meta.keys())  # already in order
        self.lock = mp.Lock()
        self.producer_cur_idx = mp.Value("i", -1)
        self.consumer_cur_idx = mp.Value("i", 0)
        # to avoid deadlock when torch.Tensor.float() is called ???
        # ref: https://github.com/pytorch/pytorch/issues/89693
        torch.set_num_threads(1)

        def _val_reader_worker_func(all_tomo_ids, producer_cur_idx, consumer_cur_idx, prefetch, lock: mp.Lock, worker_id=None, stop_flag: mp.Event = None):  # type: ignore
            if io_backend == "cv2_seq":
                tomo_reader = OpencvTomogramLoader()
            elif io_backend == "cv2":
                # multithread code should go after fork, so not impl in __init__()
                tomo_reader = MultithreadOpencvTomogramLoader(num_workers=8)
            else:
                tomo_reader = None

            keep_tomos = []
            while True:
                if stop_flag.is_set():
                    logger.info(
                        "[FAST VAL LOADER worker %d] Stop fast val loader, unlink %d remaining tomos",
                        worker_id,
                        len(keep_tomos),
                    )
                    for tomo_idx, tomo_id, shm_tomo, shm_kpts in keep_tomos:
                        shm_tomo.unlink()
                        shm_kpts.unlink()
                        logger.debug(
                            "[FAST VAL LOADER worker %d] UNLINK tomo: idx=%d id=%s shape=%s dtype=%s kpts=%s",
                            worker_id,
                            tomo_idx,
                            tomo_id,
                            shm_tomo.shape,
                            shm_tomo.dtype,
                            shm_kpts.shape,
                        )
                    if self.io_backend != "cache":
                        del tomo_reader
                    gc.collect()
                    return
                # main job
                lock.acquire()
                tomo_idx = producer_cur_idx.value + 1
                if (
                    tomo_idx < len(all_tomo_ids)
                    and producer_cur_idx.value < consumer_cur_idx.value + prefetch
                ):
                    # fetch new item
                    producer_cur_idx.value = tomo_idx
                    logger.debug(
                        "[Worker %d] producer=%s consumer=%s",
                        worker_id,
                        producer_cur_idx.value,
                        consumer_cur_idx.value,
                    )
                    lock.release()

                    tomo_id = all_tomo_ids[tomo_idx]
                    tomo_meta = all_tomos_meta[tomo_id]
                    ori_spacing = tomo_meta["ori_spacing"]
                    if "target_spacing" in tomo_meta:
                        target_spacing = tomo_meta["target_spacing"]
                    else:
                        target_spacing = self.target_spacing
                    tomo_shape = tomo_meta["tomo_shape"]
                    ori_kpts = tomo_meta["ori_kpts"]

                    if self.io_backend == "cache":
                        tomo, kpts = load_spaced_val_sample_cache(
                            tomo_id,
                            tomo_shape,
                            data_dir,
                            ori_kpts,
                            ori_spacing,
                            target_spacing,
                            self.interpolation_mode,
                        )
                    else:
                        tomo, kpts = load_spaced_val_sample(
                            tomo_id,
                            tomo_shape,
                            data_dir,
                            ori_kpts,
                            ori_spacing,
                            target_spacing,
                            transform,
                            io_backend,
                            tomo_reader,
                            interpolation_mode=self.interpolation_mode,
                        )
                    logger.debug(
                        "[Worker %d] loaded tomo_idx=%d tomo_id=%s shape=%s dtype=%s kpts=%s",
                        worker_id,
                        tomo_idx,
                        tomo_id,
                        tomo.shape,
                        tomo.dtype,
                        kpts.shape,
                    )
                    shm_tomo = ShmTensor.from_tensor(
                        tomo, f"{shm_prefix}_tomo_{tomo_id}"
                    )
                    # to prevent zero keypoints case, shape (0, 10)
                    # dirty hack: set last element (keypoint class) to -100 to indicate zero keypoint
                    if kpts.shape[0] == 0:
                        _kpts = torch.full((1, 10), -100, dtype=torch.float32)
                    else:
                        _kpts = kpts
                    shm_kpts = ShmTensor.from_tensor(
                        _kpts, f"{shm_prefix}_kpts_{tomo_id}"
                    )
                    keep_tomos.append((tomo_idx, tomo_id, shm_tomo, shm_kpts))

                    # unlink the eldest shared memory
                    rm_tomos = [e for e in keep_tomos if e[0] <= tomo_idx - keep]
                    keep_tomos = [e for e in keep_tomos if e[0] > tomo_idx - keep]
                    for tomo_idx, tomo_id, shm_tomo, shm_kpts in rm_tomos:
                        shm_tomo.unlink()
                        shm_kpts.unlink()
                else:
                    lock.release()
                    time.sleep(0.5)

        self.val_reader_worker_func = _val_reader_worker_func
        self.workers = []

    def is_running(self):
        return len(self.workers) > 0

    def start(self):
        """
        Should be called in each start of validation stage
        """
        logger.info("[FAST VAL LOADER] Starting..")
        with self.lock:
            self.producer_cur_idx.value = -1
            self.consumer_cur_idx.value = 0
        workers = []
        for worker_id in range(self.num_workers):
            worker = StoppableProcess(
                group=None,
                target=self.val_reader_worker_func,
                name=f"{self.shm_prefix}_val_reader_{worker_id}",
                args=(
                    self.all_tomo_ids,
                    self.producer_cur_idx,
                    self.consumer_cur_idx,
                    self.prefetch,
                    self.lock,
                    worker_id,
                ),
                kwargs={},
                daemon=True,
            )
            worker.start()
            workers.append(worker)
        self.workers = workers
        return workers

    def shutdown(self):
        """
        Should be called in each end of validation stage
        """
        logger.info("[FAST VAL LOADER] Shuting down..")
        with self.lock:
            self.producer_cur_idx.value = -1
            self.consumer_cur_idx.value = 0
        # stop all workers
        for worker in self.workers:
            worker.stop()

        alive_states = [True]
        while True:
            alive_states = [worker.is_alive() for worker in self.workers]
            if any(alive_states):
                logger.info(
                    "[FAST VAL LOADER] Stop but still alive with states %s",
                    alive_states,
                )
                time.sleep(1)
            else:
                logger.info(
                    "[FAST VAL LOADER] All workers stopped !",
                )
                break
        self.workers = []

    def restart(self):
        logger.info("[FAST VAL LOADER] Restarting..")
        self.shutdown()
        self.start()

    def load(self, tomo_id):
        """
        Load prefetched tomogram tensor from shared memory
        """
        tomo_idx = self.all_tomo_ids.index(tomo_id)
        with self.lock:
            self.consumer_cur_idx.value = max(self.consumer_cur_idx.value, tomo_idx)
        tomo_meta = self.all_tomo_metas[tomo_id]
        tomo_shape = tomo_meta["tomo_shape"]
        kpts_shape = tomo_meta["ori_kpts"].shape
        while True:
            if tomo_idx > self.producer_cur_idx.value:
                time.sleep(0.5)
                continue
            try:
                tomo_shm, tomo = ShmTensor(
                    shm_name=f"{self.shm_prefix}_tomo_{tomo_id}",
                    shape=tomo_shape,
                    dtype=torch.uint8,
                    create=False,
                ).to_tensor()
                if kpts_shape[0] > 0:
                    kpts_shm, kpts = ShmTensor(
                        f"{self.shm_prefix}_kpts_{tomo_id}",
                        kpts_shape,
                        torch.float32,
                        create=False,
                    ).to_tensor()
                else:
                    kpts_shm, kpts = ShmTensor(
                        f"{self.shm_prefix}_kpts_{tomo_id}",
                        (1, 10),
                        torch.float32,
                        create=False,
                    ).to_tensor()
                    # shared memory prevent zero size buffer
                    # dirty hack: last element (keypoint class) is -100 -> indicate zero keypoint
                    assert kpts[0, -1] == -100
                    kpts = torch.zeros((0, 10), dtype=torch.float32)
                return tomo, kpts, tomo_shm, kpts_shm
            except Exception as e:
                # print("exception:", e)
                time.sleep(0.5)
