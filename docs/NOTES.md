### Agg heatmap stride
```
1 -> 95.62  0.897207 (_self_ 0) │ 0.962145 (_self_ 0)
4 -> 0.9522  0.890402 (_self_ 0) │ 0.958466 (_self_ 0)
8 -> 0.955236 0.857424 (_self_ 0) │ 0.958466 (_self_ 0)
16 -> 0.963665 0.766223 (_self_ 0) │ 0.971787 (_self_ 0)
32 -> 0.926944 0.646802 (_self_ 0) │ 0.934579 (_self_ 0)
```

### Public LB results
**Model 1**
- 2 TTA -> 8 TTA: 50.6 -> 51.7
- Spacing: target_spacing=32.0, thres ~ 0.09, 2xTTA
  - 8 -> 39.6 -> 49.3
  - 13.1 -> 69.5 -> 72.7
  - 15.6 -> 50.6 -> 57.4
  - 19.7 -> 38.4 -> 44.4
- Threshold: 2xTTA, spacing 15.6 -> 32.0
  - 0.09 -> 50.6 -> 58.8
  - 0.075 -> 57.3 -> 62.1
  - 0.05 -> 59.8 -> 64.0
  - 0.01 -> 68.1 -> 65.3
- Threshold: 2xTTA, spacing 13.1 -> 32.0
  - 0.07: 70.9 -> 72.3
  - 0.06: 71.2 -> 72.5
  - 0.05 -> 71.8 -> 71.0
  - 0.04 -> 71.8 -> 71.0
  - 0.03 -> 69.3 -> 67.9
  - 0.01 -> 63.6 -> 61.0
- Spacing: ? -> 32.0, 2xTTA, thres 0.01
  - 8.0 -> 42.3 -> 49.8
  - 10.0 -> 57.3 -> 62.1
  - 12.0 -> 61.8 -> 61.6
  - 13.1 -> 63.6 -> 61.0
=> WE NEED HIGH, BUT JUST ENOUGH RESOLUTION

**Model 2**
- Threshold: heatmap_stride=16, 8xTTA, spacing 13.1 -> 16.0
  - 0.4 -> ? -> 72.8
  - 0.35 -> 60.9 -> 74.4
  - 0.30 -> 63.9 -> 75.6
  - 0.25 -> 63.6 -> 74.8
  - 0.20 -> 64.1 -> 74.0
  - 0.15 -> 62.1 -> 71.1
  - 0.10 -> 60.8 -> 66.0
- Heatmap_stride: 8xTTA, spacing 13.1 -> 16.0, thres 0.2
  - 1: 70.1 -> 76.3, + blur -> 77.3
  - 2: ?
  - 4: 70.7 -> 77.1
  - 8: 70.8 -> 77.3
  - 16: 69.4 -> 74.5


**Model 5**: `5 folds, task.decode.heatmap_stride=8, spacing 13.1 -> 16.0`
  - spacing 24: thres 0.25 -> 70.0, 0.2 -> 71.0, 0.15 -> 73.2, 0.1 -> 73.1, 0.05 -> 72.3
  - spacing 19.7: thres 0.2 -> 79.7 , thres 0.15 -> 77.5
  - spacing 16.0: thres 0.3 -> 78.8, 0.25 -> 80.1, 0.2 -> 79.8, 0.15 -> 77.8
  - spacing 13.1: thres 0.25 -> 75.0, 0.2 -> 76.7, 0.15 -> 77.3, 0.1 -> ???

- WTF: change interpolation method from trilinear -> nearest: 77.3 -> 76.1 (confirmed no bug)
- trilinear -> area: 77.3 -> 76.3 => interpolation mode really matter, even for target_spacing=16.
- Maybe, the private data voxel spacing is high (sparse), and 13.1 -> 16.0 can loose many details with nearest interpolation.
- Combined with public discussions, seem like the private data has higher resolution (1300-1500 in 90%) and higher voxel spacing so global object scale is not change too much. After transform to target spacing of 16, the real spacing is > 16, indicate performance degration with heatmap_stride=16.
- Blur improve performance


### Effect of Gaussian Blur postprocessing
no blur
```
{'Fbeta': 0.9713375796178343, 'Precision': 0.9838709677419355, 'Recall': 0.9682539682539683, 'thres': 0.51708984375, 'AP': 0.9663522687481183, 'PAP': 0.9227564267495092, 'bestR': 0.9682539682539683, 'mAP': 0.9061523894548296, 'mPAP': 0.8226269455384199, 'kaggleFbeta': 0.9713375796178344}
```

blur 0.1
```
{'Fbeta': 0.9584664536741213, 'Precision': 0.9836065573770492, 'Recall': 0.9523809523809523, 'thres': 0.32763671875, 'AP': 0.9521185705611934, 'PAP': 0.8990335964379678, 'bestR': 0.9523809523809523, 'mAP': 0.9378282701619248, 'mPAP': 0.8752164291058534, 'kaggleFbeta': 0.9584664536741214}
```

blur 0.2
```
{'Fbeta': 0.9651898734177214, 'Precision': 0.953125, 'Recall': 0.9682539682539683, 'thres': 0.172607421875, 'AP': 0.9675040154950867, 'PAP': 0.9246760046611235, 'bestR': 0.9682539682539683, 'mAP': 0.9500279821350404, 'mPAP': 0.8929037797488771, 'kaggleFbeta': 0.9651898734177216}
```

blur 0.3
```
{'Fbeta': 0.9651898734177214, 'Precision': 0.953125, 'Recall': 0.9682539682539683, 'thres': 0.0892333984375, 'AP': 0.9667034880319805, 'PAP': 0.9233417922226136, 'bestR': 0.9682539682539683, 'mAP': 0.9492754374473995, 'mPAP': 0.8942950412483117, 'kaggleFbeta': 0.9651898734177216}
```

blur 0.2 + trilinear
```
{'Fbeta': 0.9615384615384615, 'Precision': 1.0, 'Recall': 0.9523809523809523, 'thres': 0.1864013671875, 'AP': 0.9523809523809521, 'PAP': 0.8994708994708993, 'bestR': 0.9523809523809523, 'mAP': 0.9396825396825393, 'mPAP': 0.8783068783068781, 'kaggleFbeta': 0.9615384615384616}
```

## CC3D parameters (with segmentation target)

```
NMS

{'Fbeta': 0.9455128205128206, 'Precision': 0.9833333333333333, 'Recall': 0.9365079365079365, 'thres': 0.97412109375, 'AP': 0.9495588013480096, 'PAP': 0.23339165304033346, 'bestR': 0.9523809523809523, 'mAP': 0.6014745884825936, 'mPAP': 0.13737828239162417, 'kaggleFbeta': 0.9455128205128205}
```

```
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.25
DECODE_CC3D_CONF_MODE = "avg"

{'Fbeta': 0.9455128205128206, 'Precision': 0.9833333333333333, 'Recall': 0.9365079365079365, 'thres': 0.373046875, 'AP': 0.9359697908645892, 'PAP': 0.8721189636102944, 'bestR': 0.9365079365079365, 'mAP': 0.9043847466904159, 'mPAP': 0.8194772233200054, 'kaggleFbeta': 0.9455128205128205}
```

```
DECODE_CC3D_CONF_THRES = 0.05
DECODE_CC3D_RADIUS_FACTOR = 0.1
DECODE_CC3D_CONF_MODE = "avg"

{'Fbeta': 0.9455128205128206, 'Precision': 0.9833333333333333, 'Recall': 0.9365079365079365, 'thres': 0.373046875, 'AP': 0.9359697908645892, 'PAP': 0.8721189636102944, 'bestR': 0.9365079365079365, 'mAP': 0.9043847466904159, 'mPAP': 0.8194772233200054, 'kaggleFbeta': 0.9455128205128205}
```