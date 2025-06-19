

```
❯ tree -L 4 ./
./
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
- `processed/pseudo_label/`: additional detailed data, contains images to upload to CVAT and refined annotations downloaded from CVAT to create the final refined annotations. The annotations refinement process contains 3 rounds: `round 1` (refine train data), `round 2` (refine train data), `external` (refine external CryoET Portal data)