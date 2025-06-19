"""
Some archived models zoo to tracks some trained models and results during early stage of this competition, which is currently not useful anymore.
"""

##################################################
###### MODEL SIGNATURES
##################################################

SIG_0_0328_x3d = {
    "model": "X3D",
    "heatmap_stride": 16,
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy"],
    "torch_combined_weight_path": "0_0328_x3d/model.pth",
    "trt_combined_weight_path": "0_0328_x3d/model.engine",
    "onnx_combined_weight_path": "0_0328_x3d/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "batch_size": 1,
    "threshold": 0.0911865,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "0_0328_x3d/config.yaml",
            "ckpt_path": "0_0328_x3d/ep=2_step=8700_val_Fbeta=0.917405_val_PAP=0.850334.ckpt",
            "ema": 0.99,
        }
    ],
}

SIG_1_0330_x3d = {
    "model": "X3D",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": [
    #     "zyx",
    #     "zyx_x",
    #     "zyx_y",
    #     "zyx_z",
    #     "zyx_xy",
    #     "zyx_xz",
    #     "zyx_yz",
    #     "zyx_xyz",
    #     "zxy",
    #     "zxy_x",
    #     "zxy_y",
    #     "zxy_z",
    #     "zxy_xy",
    #     "zxy_xz",
    #     "zxy_yz",
    #     "zxy_xyz",
    # ],
    "torch_combined_weight_path": "1_0330_x3d/model.pth",
    "trt_combined_weight_path": "1_0330_x3d/model.engine",
    "onnx_combined_weight_path": "1_0330_x3d/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "batch_size": 1,
    "threshold": 0.30712890625,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "1_0330_x3d/config.yaml",
            "ckpt_path": "1_0330_x3d/ep=3_step=13500_val_Fbeta=0.958958_val_PAP=0.972517.ckpt",
            "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        }
    ],
}

SIG_2_0405_r50tsn = {
    "model": "R50TSN",
    "heatmap_stride": 16,
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": [
    #     "zyx",
    #     "zyx_x",
    #     "zyx_y",
    #     "zyx_z",
    #     "zyx_xy",
    #     "zyx_xz",
    #     "zyx_yz",
    #     "zyx_xyz",
    #     "zxy",
    #     "zxy_x",
    #     "zxy_y",
    #     "zxy_z",
    #     "zxy_xy",
    #     "zxy_xz",
    #     "zxy_yz",
    #     "zxy_xyz",
    # ],
    "torch_combined_weight_path": "2_0405_r50tsn/model.pth",
    "trt_combined_weight_path": "2_0405_r50tsn/model.engine",
    "onnx_combined_weight_path": "2_0405_r50tsn/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [192, 224, 224],
    "batch_size": 1,
    "threshold": 0.257568,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "2_0405_r50tsn/config.yaml",
            "ckpt_path": "2_0405_r50tsn/ep=3_step=10000_val_Fbeta=0.922665_val_PAP=0.864329.ckpt",
            "ema": 0.99,
        }
    ],
}

SIG_3_0905_x3d_spacing64 = {
    "model": "X3D",
    "heatmap_stride": 4,
    # "tta": ["zyx"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    "tta": [
        "zyx",
        "zyx_x",
        "zyx_y",
        "zyx_z",
        "zyx_xy",
        "zyx_xz",
        "zyx_yz",
        "zyx_xyz",
        "zxy",
        "zxy_x",
        "zxy_y",
        "zxy_z",
        "zxy_xy",
        "zxy_xz",
        "zxy_yz",
        "zxy_xyz",
    ],
    "torch_combined_weight_path": "3_0905_x3d_spacing64/model.pth",
    "trt_combined_weight_path": "3_0905_x3d_spacing64/model.engine",
    "onnx_combined_weight_path": "3_0905_x3d_spacing64/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [256, 320, 320],
    "batch_size": 1,
    "threshold": 0.05133056640625,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "3_0905_x3d_spacing64/config.yaml",
            "ckpt_path": "3_0905_x3d_spacing64/ep=8_step=10000_val_Fbeta=0.630734_val_PAP=0.255650.ckpt",
            "ema": 0.99,
        }
    ],
}


SIG_5_0416_x3d_spacing16_5folds = {
    "model": "X3D",
    "heatmap_stride": 16,
    # "tta": ["zyx"],
    "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": ["zyx",  "zyx_x", "zyx_y","zyx_z", "zyx_xy", "zyx_xz", "zyx_yz", "zyx_xyz", "zxy", "zxy_x", "zxy_y", "zxy_z", "zxy_xy", "zxy_xz", "zxy_yz", "zxy_xyz"],
    "torch_combined_weight_path": "5_0416_x3d_spacing16_5folds/model.pth",
    "trt_combined_weight_path": "5_0416_x3d_spacing16_5folds/model.engine",
    "onnx_combined_weight_path": "5_0416_x3d_spacing16_5folds/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "5_0416_x3d_spacing16_5folds/config.yaml",
            "ckpt_path": "5_0416_x3d_spacing16_5folds/fold0_ep=2_step=9000_val_Fbeta=0.948678_val_PAP=0.929133.ckpt",
            "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        },
        # {
        #     "fold": 1,
        #     "config_path": "5_0416_x3d_spacing16_5folds/config.yaml",
        #     "ckpt_path": "5_0416_x3d_spacing16_5folds/fold1_ep=1_step=8000_val_Fbeta=0.943897_val_PAP=0.895131.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 2,
        #     "config_path": "5_0416_x3d_spacing16_5folds/config.yaml",
        #     "ckpt_path": "5_0416_x3d_spacing16_5folds/fold2_ep=2_step=12000_val_Fbeta=0.949074_val_PAP=0.921348.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 3,
        #     "config_path": "5_0416_x3d_spacing16_5folds/config.yaml",
        #     "ckpt_path": "5_0416_x3d_spacing16_5folds/fold3_ep=3_step=14000_val_Fbeta=0.918210_val_PAP=0.856209.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 4,
        #     "config_path": "5_0416_x3d_spacing16_5folds/config.yaml",
        #     "ckpt_path": "5_0416_x3d_spacing16_5folds/fold4_ep=2_step=11000_val_Fbeta=0.868056_val_PAP=0.804776.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
    ],
}


SIG_6_5folds_spacing16_bg02 = {
    "model": "X3D",
    "heatmap_stride": 16,
    "tta": ["zyx"],
    # "tta": ["zyx", "zxy"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y"],
    # "tta": ["zyx", "zxy", "zyx_x", "zyx_y", "zxy_x", "zxy_y", "zyx_z", "zxy_z"],
    # "tta": [
    #     "zyx",
    #     "zyx_x",
    #     "zyx_y",
    #     "zyx_z",
    #     "zyx_xy",
    #     "zyx_xz",
    #     "zyx_yz",
    #     "zyx_xyz",
    #     "zxy",
    #     "zxy_x",
    #     "zxy_y",
    #     "zxy_z",
    #     "zxy_xy",
    #     "zxy_xz",
    #     "zxy_yz",
    #     "zxy_xyz",
    # ],
    "torch_combined_weight_path": "6_5folds_spacing16_bg0.2-0.01/model.pth",
    "trt_combined_weight_path": "6_5folds_spacing16_bg0.2-0.01/model.engine",
    "onnx_combined_weight_path": "6_5folds_spacing16_bg0.2-0.01/model.onnx",
    "trt_precision": "fp16",
    "patch_size": [224, 448, 448],
    "border": [0, 0, 0],
    "overlap": [0, 0, 0],
    "batch_size": 1,
    "threshold": 0.123456789,
    "fold_metas": [
        {
            "fold": 0,
            "config_path": "6_5folds_spacing16_bg0.2-0.01/config.yaml",
            "ckpt_path": "6_5folds_spacing16_bg0.2-0.01/fold0_ep=2_step=12001_val_Fbeta=0.961838_val_PAP=0.918572.ckpt",
            "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        },
        # {
        #     "fold": 1,
        #     "config_path": "6_5folds_spacing16_bg0.2-0.01/config.yaml",
        #     "ckpt_path": "6_5folds_spacing16_bg0.2-0.01/fold1_ep=2_step=13000_val_Fbeta=0.938450_val_PAP=0.901499.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 2,
        #     "config_path": "6_5folds_spacing16_bg0.2-0.01/config.yaml",
        #     "ckpt_path": "6_5folds_spacing16_bg0.2-0.01/fold2_ep=0_step=4000_val_Fbeta=0.961089_val_PAP=0.960101.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 3,
        #     "config_path": "6_5folds_spacing16_bg0.2-0.01/config.yaml",
        #     "ckpt_path": "6_5folds_spacing16_bg0.2-0.01/fold3_ep=2_step=14001_val_Fbeta=0.941358_val_PAP=0.887824.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
        # {
        #     "fold": 4,
        #     "config_path": "6_5folds_spacing16_bg0.2-0.01/config.yaml",
        #     "ckpt_path": "6_5folds_spacing16_bg0.2-0.01/fold4_ep=1_step=12000_val_Fbeta=0.925362_val_PAP=0.890058.ckpt",
        #     "ema": 0.99,  # _self_ has best Fbeta, but ema has better PAP/mPAP
        # },
    ],
}
