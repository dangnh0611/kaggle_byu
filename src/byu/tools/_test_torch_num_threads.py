import torch
import multiprocessing as mp
from monai.data.meta_tensor import MetaTensor
import numpy as np


def worker():
    print("INSIDE NUM THREADS:", torch.get_num_threads())
    torch.set_num_threads(32)
    arr = torch.zeros((1500, 1200, 900), dtype=torch.uint8)
    arr = MetaTensor(arr)
    print(type(arr), arr.shape, arr.dtype, arr.device)
    arr2 = arr.to(
        torch.float32,
    )
    print(type(arr2), arr2.shape, arr2.dtype, arr2.device)


# print('OUTSIDE NUM THREADS:', torch.get_num_threads())
# worker = mp.Process(None, worker)
# worker.start()
# worker.join()
# print('ALL DONE!')


import torch
import torch.utils.data as data
import os
from byu.data.io import MultithreadOpencvTomogramLoader


class DummyDataset(data.Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        print(f"PID: {os.getpid()}, Threads: {torch.get_num_threads()}")

        if getattr(self, "tomo_loader", None) is None:
            self.tomo_loader = MultithreadOpencvTomogramLoader(num_workers=8)

        arr = self.tomo_loader.load("/home/dangnh36/datasets/byu/raw/train/tomo_fea6e8")
        arr = torch.from_numpy(arr)
        arr = MetaTensor(arr, meta={"a": 1, "b": float(2.123)})
        print(type(arr), arr.shape, arr.dtype, arr.device)
        arr2 = arr.to(
            dtype=torch.float32, device=None, memory_format=torch.contiguous_format
        )
        print(type(arr2), arr2.shape, arr2.dtype, arr2.device)
        return arr2


def worker_init_fn(worker_id):
    torch.set_num_threads(7)  # Set to 4 threads per worker


dataset = DummyDataset()

dataloader = torch.utils.data.DataLoader(
    dataset, num_workers=2, worker_init_fn=worker_init_fn
)

# Iterate to trigger worker processes
for _ in dataloader:
    print(_.shape)
    pass
