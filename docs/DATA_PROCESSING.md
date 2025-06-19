This file explains in details about data processing and how each file in [this dataset](https://www.kaggle.com/datasets/dangnh0611/byu-processed-data) was created.

## Table Of Contents
- [Table Of Contents](#table-of-contents)
- [Overview](#overview)
- [External Data](#external-data)
- [Pseudo Labeling](#pseudo-labeling)
  - [Pseudo Labeling on official competition dataset round 1](#pseudo-labeling-on-official-competition-dataset-round-1)
  - [Pseudo Labeling on official competition dataset round 2](#pseudo-labeling-on-official-competition-dataset-round-2)
  - [Pseudo Labeling on external dataset](#pseudo-labeling-on-external-dataset)


## Overview

```
❯ tree -L 4 ./data
./data
|-- README.md
|-- external
|   |-- brendanartley_external_labels.csv
|   |-- external_labels_v1.csv
|   |-- meta
|   |   |-- README.md
|   |   `-- pwd_structure.txt
|   |-- metadata_full.csv
|   |-- tatamikenn_irregular_labels.csv
|   `-- tomogram
|       |-- README.md
|       `-- pwd_structure.txt
`-- processed
    |-- all_gt.csv
    |-- all_gt_v3.csv
    |-- cv
    |   |-- v1
    |   |   |-- skf4_rd42.json
    |   |   `-- skf5_rd42.json
    |   |-- v2
    |   |   |-- skf4_rd42.json
    |   |   `-- skf5_rd42.json
    |   `-- v3
    |       `-- skf4_rd42.json
    |-- external_gt.csv
    |-- gt.csv
    |-- gt_v2.csv
    |-- gt_v3.csv
    |-- histogram.json
    |-- pseudo_label
    |   |-- external
    |   |   |-- FN
    |   |   |-- FP
    |   |   |-- cvat.xml
    |   |   `-- meta.json
    |   |-- round1
    |   |   |-- FN
    |   |   |-- FP
    |   |   |-- annotations.xml
    |   |   |-- fold0.csv
    |   |   |-- fold1.csv
    |   |   |-- fold2.csv
    |   |   |-- fold3.csv
    |   |   `-- fold4.csv
    |   `-- round2
    |       |-- FN
    |       |-- FP
    |       |-- cvat.xml
    |       `-- meta.json
    `-- train_labels_v2.csv

18 directories, 32 files
```

Details of important files/directories:
- `external/brendanartley_external_labels.csv`: `labels.csv` file downloaded from https://www.kaggle.com/datasets/brendanartley/cryoet-flagellar-motors-dataset
- `external/tatamikenn_irregular_labels.csv`: `irregular_labels.csv` file downloaded from https://www.kaggle.com/datasets/tatamikenn/byu-cryoet-dataset-with-pixel-anomalies-corrected/data
- `external/external_labels_v1.csv`: external dataset's annotations, converted from @brendanartley 's annotations, compatible with competition data's format
- `external/meta`: directory contains additional metadata files for each tomogram downloaded from CryoET Data Portal
- `external/tomogram`: directory contains downloaded tomograms from CryoET Data Portal, compatible with competition data's format
- `external/metadata_full.csv`: contains all metadata for all external tomograms, combined/derived from `external/meta`
- `processed/cv`: exported metadata for Cross-Validation, simply using StratifiedKFold, take a look at `notebooks/cv_split.ipynb` for more details
- `processed/gt.csv`: original competition's groundtruth in another format, using only one row for each tomogram
- `processed/gt_v2.csv`: refined competition's groundtruth after round 1 (pseudo labeling + manual review)
- `processed/gt_v3.csv`: refined competition's groundtruth after round 2 (pseudo labeling + manual review)
- `processed/external_gt.csv`: refined external dataset's groundtruth (pseudo labeling + manual review)
- `processed/all_gt.csv`: concatenation of `processed/gt_v2.csv` + `processed/external_gt.csv`
- `processed/all_gt_v3.csv`: concatenation of `processed/gt_v3.csv` + `processed/external_gt.csv`
- `processed/pseudo_label/`: additional supplemental data, contains images to upload to CVAT and refined annotations downloaded from CVAT to create the final refined annotations. The annotations refinement process contains 3 rounds: `round 1` (refine official train data), `round 2` (refine official train data), `external` (refine external CryoET Portal data)

## External Data
I already provided some lightweight metadata and groundtruth files in [../data/](../data/) directory. However, you still need to download the external tomograms data. External dataset's tomograms can be download from CryoET Data Portal using [this script](../src/byu/tools/download_external_dataset.py), this should take up additional 180GB of your disk space:
```bash
cd ${THIS_REPO_ROOT_DIR}
# You can change number of download workers and number of processing workers suited for your local environment
# Larger number of processing workers can swallow up your RAM, be careful :)
python3 src/byu/tools/download_external_dataset.py --output-dir data/external/ --tmp-download-dir data/external/tmp/ --num-download-workers=8 --num-process-workers=1
# clear the temporary downloaded files if any
rm -rf data/external/tmp/
```
Running the above script will populate the [../data/external/meta/](../data/external/meta/) and [../data/external/tomogram/](../data/external/tomogram/) directory


## Pseudo Labeling
Since pseudo-labeling followed by human review is a complex, multi-stage process and **not fully automated**, this guide will provide only a high-level overview and brief description of how the pseudo-labels were generated and refined.  
**You can also reproduce this process yourself, but will need to modify some paths/configs in some scripts/notebooks to make theme suited your need.**

### Pseudo Labeling on official competition dataset round 1
Steps:  
1. Train a 5-folds 3D-UNet model with X3D-M encoder on official competition data
2. Use each trained model for each fold to predict on the corresponding fold's validation samples to prevent data leakage. I just consider samples with less than 1 motor only, which means samples with more than 1 motor will not be refined. Use a small confident threshold of 0.05, and allow to output multiple motors candidates per tomogram. This process use [../src/byu/inference/labeling_train_round1.py](../src/byu/inference/labeling_train_round1.py) script, e.g
   ```
   python3 src/byu/inference/labeling_train_round1.py --fold 0
   python3 src/byu/inference/labeling_train_round1.py --fold 1
   python3 src/byu/inference/labeling_train_round1.py --fold 2
   python3 src/byu/inference/labeling_train_round1.py --fold 3
   python3 src/byu/inference/labeling_train_round1.py --fold 4
   ```
3. Upload False Negative (FN) and False Positive (FP) cases to CVAT for manual reviewing/annotating as a simple tagging task.
4. Download the annotations and patch a refined version of official train dataset. Checkout [../notebooks/create_train_pseudo_label_round1.ipynb](../notebooks/create_train_pseudo_label_round1.ipynb) for more details.
5. This results in 2 additional files created:
   - `../data/processed/train_labels_v2.csv`: Refined + same format with compeition annotations
   - `../data/processed/gt_v2.csv`: refined competition's annotations after this round 1

### Pseudo Labeling on official competition dataset round 2
Steps:
1. Train three 3D-UNet models with encoders: X3D-M, 3D Resnet101, 2.5D ConvNeXt Tiny. Ensemble of these models scored 0.852 on Private LB and 0.850 on Public LB, also provided in [version 262 of my submission notebook](https://www.kaggle.com/code/dangnh0611/3rd-place-solution-submit?scriptVersionId=240301171).
2. Use ensemble of those 3 models, each with 5 TTAs to predict on all samples of official competition dataset. This cause train-test leakage, but I noticed just a small gap between train/validation score (caused by underfiting or label noise), so leakage is still okay for me. This process use [../src/byu/inference/labeling_train_round2.py](../src/byu/inference/labeling_train_round2.py) script, e.g
    ```
    python3 src/byu/inference/labeling_train_round2.py
    ```
3. Upload False Negative (FN) and False Positive (FP) cases to CVAT for manual reviewing/annotating as a simple tagging task.
4. Download the annotations and patch a refined version of official train dataset. Checkout [../notebooks/create_train_pseudo_label_round2.ipynb](../notebooks/create_train_pseudo_label_round2.ipynb) for more details.
5. This results in 1 additional files created:
   - `../data/processed/gt_v3.csv`: refined competition's annotations after this round 2

### Pseudo Labeling on external dataset
Nearly the same process as in the above [Pseudo labeling on official competition dataset round 2](#pseudo-labeling-on-official-competition-dataset-round-2) section.

Steps:
1. Train three 3D-UNet models with encoders: X3D-M, 3D Resnet101, 2.5D ConvNeXt Tiny. Ensemble of these models scored 0.852 on Private LB and 0.850 on Public LB, also provided in [version 262 of my submission notebook](https://www.kaggle.com/code/dangnh0611/3rd-place-solution-submit?scriptVersionId=240301171).
2. Use ensemble of those 3 models, each with 5 TTAs to predict on all samples of external CryoET Data Portal dataset. This process use [../src/byu/inference/labeling_external.py](../src/byu/inference/labeling_external.py) script, e.g
    ```
    python3 src/byu/inference/labeling_external.py
    ```
3. Upload False Negative (FN) and False Positive (FP) cases to CVAT for manual reviewing/annotating as a simple tagging task.
4. Download the annotations and patch a refined version of the external dataset. Checkout [../notebooks/create_external_pseudo_label.ipynb](../notebooks/create_external_pseudo_label.ipynb) for more details.
5. This results in 3 additional files created:
   - `../data/processed/external_gt.csv`: refined external dataset's annotations
   - `../data/processed/all_gt.csv`: concatenation of `../data/processed/gt_v2.csv` + `../data/processed/external_gt.csv`
   - `../data/processed/all_gt_v3.csv`: concatenation of `../data/processed/gt_v3.csv` + `../data/processed/external_gt.csv`