# BYU - Locating Bacterial Flagellar Motors 2025
https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025


## TODOs
- [x] Train a scale prediction model
- [x] Handle inversed intensity tomo
- [x] Optical Flow, or background substraction
- [x] Multiscale TTA
- [x] External data
- [x] Probe LB/PB pixel histogram/distribution
- [x] Train set has wrong label (missing detections) -> fix it
- [x] Losses:
  - [x] JSD with softmax
  - [x] BCE with higer pos weight
  - [x] Tversky
- [x] Better neck: DLA to fuse low-level features with high-level features
- [x] Deeper: larger stride than 32 + higher resolution input. Combo of RSNA Breast Cancer ? :) ConvNext should be better too :)
- [ ] Denoise: Median Filter?
- [x] Late sigmoid, ensemble of logits -> sigmoid
- [x] Encoders
  - [x] X3D
  - [x] I3D
  - [x] MedNext
  - [x] Hiera
  - [x] resnet18, resnet50, resnet101
  - [x] resnext50_32x4d, resnext101_32x8d
  - [x] densenet121, densenet201, densenet161 (shallower but wider)
  - [x] efficientnet-b0, efficientnet-b2, efficientnet-b5, efficientnet-b7
  - [x] mit_b0, mit_b
  - [x] tu-convnext_tiny.fb_in22k_ft_in1k_384, tu-convnext_small.fb_in22k_ft_in1k_384. Note: `Atto (0.5M) < Femto (1.6M) < Pico (3.9M) < Nano (7M) < Tiny (28M) < Small (50M) < Base (89M)`


## Experiments
- [x] Sweep Interpolation mode for 2D model
- [x] Train a X3DM with mixup mix+alpha=1.0
- [x] Fixed radius for Gaussian heatmap
- [x] External dataset wrong voxel spacing: 10210, 10211
- [x] Tune the eps of BatchNorm, or using smaller batch size so that Batchnorm is updated more frequently
- [x] Re-test the bad performance model on validation set
- [ ] Correct the posible quantization error on private test -> no change in LB, not sure about PB
- [x] F.interpolate with antialias=True -> 2D only
- [x] Last cooldown epochs with no augmentation, same as YOLOX so that BatchNorm worker better, reduce domain shift -> not work
- [x] Larger Z spacing, smaller XY spacing -> reduce runtime
- [x] Warmup
- [x] Loss
  - [x] MSE, MAE -> drop
  - [x] Try Tversky only
  - [x] Tune the combine weight ratio
  - [x] Hard example mining: FocalLoss
  - [x] DSNT ? need to EDA first
- [x] Too large patch size could reduce perf (due to black padding + BN?) -> [224,224,224], [224,320,320], [224,448,448], [192,512,512]. [128,768,768] not work well
- [x] Smaller heatmap stride, e.g 8 or 4
- [x] sigma (after finding best patch size) -> keep 0.2
- [x] label smoothing
- [x] tune data.sampling.pre_patch.bg_ratio
- [ ] larger data.sampling.pre_patch.overlap
- [x] sampling: rand_crop + no rand_center + no correct_center + rand_shift, tune pos/neg weight as well
- [x] target_spacing: smaller?
- [x] Random target spacing -> RandZoom did the job
- [x] data.transform.heatmap_same_std
- [x] AUGMENT params tunning:
  - [x] MIXUP
  - [x] CUTMIX
  - [ ] data.aug.intensity_prob
  - [ ] data.aug.affine2_prob
  - [ ] data.aug.zoom_prob
  - [ ] data.aug.hist_equalize
  - [ ] data.aug.downsample_prob
  
- [x] Tune the decoder: larger decoder_channels, BN or LN
- [x] Multiscale loss (low priority if stride is large)
- [x] LookAHead -> not effective, ignore
- [ ] optim.weight_decay, seem like not overfit but underfit
- [x] low priority, test other EMA decay: 0.995
- [x] Patch size: (448, 448, 224) vs (768, 768, 128)
- [x] Different kind of heatmap: gaussian, segment, point
- [x] Adaptive radius scale as in pose estimation?
- [ ] DSNT approach

## LB probing
- [x] HEATMAP_AGG_MODE = 'max' to increase Recall ? Use mean first, if can't detect any -> use max
- [x] Train with 224x448, test with smaller and larger patch size
- [x] 10 submissions probing


## Error Analysis
- Noise strongly affect interpolation result -> Denoise before spacing


### Visualization
- Even for TP, prob is low (~0.6) -> change loss function, e.g more pos weight
- FP near tomo boundary, with lower confident. Possible fixes:
  - Ignore loss near boundary
  - Multi-patches inference with border and overlap
- FN at tomo boundary -> fixes: overlap/border cut + ensemble with 2D-only model
- Boundary of object seems to be assigned higher probability on heatmap

### Exp results
- Larger target_spacing (lower resolution, smaller object scale) -> model tends to predict less positive -> threshold decrease.
  - if use larger target spacing -> try to use lower patch size to prevent padding effect

## Bug fixes
- [x] sliding with border and overlap
- [x] FactFPN GLU not GELU -> keep previous behavior
- [x] Per-dim spacing is not compatible with `heatmap_same_std`


## Probing
```
57.4 -> 0.6 -> 7
62.1 -> 0.05 -> 1
65.0 -> 0.55 -> 6
67.3 -> 0.1 -> 2
68.7 -> 0.5 -> 10
70.0 -> 0.15 -> 3
71.4 -> 0.45 -> 9
72.8 -> 0.4 -> 8
73.9 -> 0.2 -> 4
74.8 -> 0.25 -> 5
75.1 -> 0.35 !!!!!
75.1 -> 0.30 !!!!!
```

```
- pixel_mean:
  - 67.4 -> 0.2 -> 4 -> (3.5, 4.5) -> in range (117.5, 122.5)
  - 66.4 -> 5 -> (4.5, 5.5) -> (120.25, 120.75) -> lower than train
- pixel_std: 62.8 -> 0.35 -> 7 -> (6.5, 7.5) -> in range (51, 53) -> higher std
- mean_Z: 61.5 -> 0.05 -> 1 -> (0.05, 1.5) -> in range (375, 425)
- mean_Y: 45.6 -> 6 -> (5.5, 6.5) -> (1375, 1425)
- mean_X: 65.3 -> 3 -> (2.5, 3.5) -> (1025, 1075)
```