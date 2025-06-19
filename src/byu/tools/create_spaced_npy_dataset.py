import argparse
import os
import numpy as np
import multiprocessing as mp
import polars as pl
import queue
from byu.data.io import MultithreadOpencvTomogramLoader
from torch.nn import functional as F
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/home/dangnh36/datasets/byu/raw/train/",
        help="Path to the source directory",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="/home/dangnh36/datasets/byu/processed/spaced/",
        help="Path to the destination directory",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="/home/dangnh36/datasets/byu/raw/train_labels.csv",
        help="Path to the CSV label file",
    )
    parser.add_argument("--spacing", type=float, help="Spacing value (float)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of concurrent workers (multiprocessing Pool)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for torch.interpolate",
    )
    parser.add_argument(
        "--interpolation_mode",
        type=str,
        default="trilinear",
        help="torch.interpolate mode",
    )
    return parser.parse_args()


def spacing_with_torch(
    tomo, ori_spacing, target_spacing, device="cuda", mode="trilinear"
):
    assert len(tomo.shape) == 3
    tomo = torch.from_numpy(tomo[None, None]).to(device)
    if mode != "nearest":
        tomo = tomo.half()
    print(
        f"ORI: shape={tomo.shape} dtype={tomo.dtype} ori_spacing={ori_spacing} target_spacing={target_spacing} scale_factor={ori_spacing / target_spacing}"
    )

    ##### ISSUES #####
    ### with interpolation mode `nearest`
    # RuntimeError: upsample_nearest3d only supports output tensors with less than INT_MAX elements, but got [1, 1, 1178, 1367, 1367]
    # ref: https://github.com/pytorch/pytorch/issues/144855
    ### with interpolation mode `trilinear` on GPU
    # RuntimeError: CUDA error: invalid configuration argument
    ##### CURRENT FIX: if CUDA interpolate kernel fails, use CPU
    try:
        spaced_tomo = F.interpolate(
            tomo,
            size=None,
            scale_factor=ori_spacing / target_spacing,
            mode=mode,
            align_corners=None if mode == "nearest" else False,
            recompute_scale_factor=False,
        )[0, 0]
    except RuntimeError as e:
        print("EXCEPTION OCCUR: {e}\nTrying CPU interpolation..")
        spaced_tomo = F.interpolate(
            tomo.cpu(),
            size=None,
            scale_factor=ori_spacing / target_spacing,
            mode=mode,
            align_corners=None if mode == "nearest" else False,
            recompute_scale_factor=False,
        )[0, 0]

    print(
        f"DST: shape={spaced_tomo.shape} dtype={spaced_tomo.dtype} min={spaced_tomo.min().item()} max={spaced_tomo.max().item()}"
    )
    if mode != "nearest":
        spaced_tomo = spaced_tomo.to(torch.uint8)
    assert spaced_tomo.dtype == torch.uint8
    spaced_tomo = spaced_tomo.cpu().numpy()
    return spaced_tomo


def worker_func(args, in_queue, out_queue):
    target_spacing = args.spacing
    tomo_loader = MultithreadOpencvTomogramLoader(8)
    while True:
        try:
            tomo_idx, tomo_id, ori_spacing = in_queue.get(block=False)
        except queue.Empty:
            print("Empty queue, exit..")
            return

        tomo_dir = os.path.join(args.src_dir, tomo_id)

        # read ori tomo
        tomo = tomo_loader.load(tomo_dir)
        ori_shape = tomo.shape  # ZYX

        print("TOMO number", tomo_idx)
        spaced_tomo = spacing_with_torch(
            tomo, ori_spacing, target_spacing, args.device, args.interpolation_mode
        )
        spaced_shape = spaced_tomo.shape
        npy_save_path = os.path.join(
            args.dst_dir,
            f"{target_spacing}_{args.interpolation_mode}",
            f"{tomo_id}.npy",
        )
        os.makedirs(os.path.dirname(npy_save_path), exist_ok=True)
        np.save(npy_save_path, spaced_tomo)
        out_queue.put((tomo_id, ori_shape, ori_spacing, spaced_shape, target_spacing))


def main(args):
    print(args)
    df = (
        pl.scan_csv(args.label_path)
        .group_by("tomo_id")
        .first()
        .sort("tomo_id")
        .collect()
    )
    print("Number of tomos:", len(df))

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    for i, row in enumerate(df.iter_rows(named=True)):
        # if i < 230:
        #     continue
        in_queue.put([i, row["tomo_id"], row["Voxel spacing"]])

    if args.num_workers > 1:
        workers = []
        for worker_idx in range(args.num_workers):
            worker = mp.Process(
                group=None,
                target=worker_func,
                args=(args, in_queue, out_queue),
                kwargs={},
                daemon=None,
            )
            worker.start()
            workers.append(worker)
        [worker.join() for worker in workers]
    else:
        worker_func(args, in_queue, out_queue)

    ret = []
    while True:
        try:
            item = out_queue.get(False)
        except queue.Empty:
            break
        ret.append(item)
    assert len(ret) == len(df)
    col_names = [
        "tomo_id",
        "ori_shape",
        "ori_spacing",
        "target_shape",
        "target_spacing",
    ]
    assert len(col_names) == len(ret[0])
    meta_df = pl.DataFrame(
        {col_name: [str(row[i]) for row in ret] for i, col_name in enumerate(col_names)}
    ).sort("tomo_id")
    print("META DATAFRAME:\n", meta_df)
    meta_df.write_csv(
        os.path.join(args.dst_dir, f"{args.spacing}_{args.interpolation_mode}.csv")
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
