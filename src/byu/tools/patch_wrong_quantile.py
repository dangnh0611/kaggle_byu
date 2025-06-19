import os
import shutil

from tqdm import tqdm

WRONG_QUANTILE_TOMO_IDS = [
    "mba2011-02-16-1",
    "mba2011-02-16-106",
    "mba2011-02-16-108",
    "mba2011-02-16-11",
    "mba2011-02-16-116",
    "mba2011-02-16-12",
    "mba2011-02-16-123",
    "mba2011-02-16-129",
    "mba2011-02-16-133",
    "mba2011-02-16-139",
    "mba2011-02-16-141",
    "mba2011-02-16-145",
    "mba2011-02-16-153",
    "mba2011-02-16-155",
    "mba2011-02-16-157",
    "mba2011-02-16-160",
    "mba2011-02-16-162",
    "mba2011-02-16-17",
    "mba2011-02-16-176",
    "mba2011-02-16-19",
    "mba2011-02-16-26",
    "mba2011-02-16-27",
    "mba2011-02-16-28",
    "mba2011-02-16-29",
    "mba2011-02-16-32",
    "mba2011-02-16-33",
    "mba2011-02-16-34",
    "mba2011-02-16-40",
    "mba2011-02-16-42",
    "mba2011-02-16-46",
    "mba2011-02-16-48",
    "mba2011-02-16-53",
    "mba2011-02-16-55",
    "mba2011-02-16-60",
    "mba2011-02-16-64",
    "mba2011-02-16-65",
    "mba2011-02-16-67",
    "mba2011-02-16-71",
    "mba2011-02-16-75",
    "mba2011-02-16-79",
    "mba2011-02-16-88",
    "mba2011-02-16-90",
    "mba2011-02-16-95",
]
print("NUMBER OF WRONG QUANTILE TOMOGRAMS:", len(WRONG_QUANTILE_TOMO_IDS))

SRC_TOMO_DIR = "/home/dangnh36/datasets/.comp/byu/external_wrong_quantile/tomogram/"
SRC_META_DIR = "/home/dangnh36/datasets/.comp/byu/external_wrong_quantile/meta/"

DST_TOMO_DIR = "/home/dangnh36/datasets/.comp/byu/external/tomogram/"
DST_META_DIR = "/home/dangnh36/datasets/.comp/byu/external/meta/"


for tomo_id in tqdm(WRONG_QUANTILE_TOMO_IDS):
    src_tomo_dir = os.path.join(SRC_TOMO_DIR, tomo_id)
    assert os.path.isdir(src_tomo_dir)
    dst_tomo_dir = os.path.join(DST_TOMO_DIR, tomo_id)
    assert os.path.isdir(dst_tomo_dir)
    shutil.rmtree(dst_tomo_dir)
    shutil.move(src_tomo_dir, DST_TOMO_DIR)

    src_meta_path = os.path.join(SRC_META_DIR, f"{tomo_id}.json")
    assert os.path.isfile(src_meta_path)
    dst_meta_path = os.path.join(DST_META_DIR, f"{tomo_id}.json")
    os.remove(dst_meta_path)
    shutil.move(src_meta_path, DST_META_DIR)
