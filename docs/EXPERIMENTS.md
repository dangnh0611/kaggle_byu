## FINAL EXPERIMENTS

### EXP1_X3DM
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM-ALLDATA-SEED42 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```

### EXP2_X3DL
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DL-ALLDATA-SEED42 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False model.encoder.model_size=L model.encoder.pretrained=ckpts/x3d_l.pyth trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```

### EXP3_X3DL_COOLDOWNFINETUNE
```bash
python3 -m yagm.run -m local=local cv.train_on_all=False 'cv.fold_idx=0' exp_name=COOLDOWNFINETUNE_UNET3D_X3DM-LE1-SEED42 exp=3d_unet_x3d optim.lr=5e-5 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=8000 scheduler.warmup_steps=0 trainer.val_check_interval=1000 'callbacks.validation_scheduler.milestones=[1000000]' data.filter_rule=le1 'ckpt.path="ckpts/ALPHA=1_ep=3_step=20000_val_Fbeta=0.943344_val_PAP=0.921336.ckpt"' data.aug.enable=False
```


### EXP4_R101
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=SWEEPLR-UNET3D_R101-ALLDATA-SEED42 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnet101 trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP5_R50
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_R50-ALLDATA-SEED42 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnet50 trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```


### EXP6_X3DM_MIXUPMIX1
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=MIXUP_MIX1.0-UNET3D_X3DM-ALLDATA-SEED42 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.aug.mixup_prob=0.2 data.aug.mixup_target_mode=mix data.aug.mixer_alpha=1.0
```


### EXP7_DENSENET121
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=SWEEPLR-UNET3D_DENSENET121-ALLDATA-SEED611 seed=611 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=densenet121 trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP8_CONVNEXT_AVG
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=UNET3D_CONVNEXTTINY_LSTM-ALLDATA-SEED611-LR5e-5 seed=611 exp=3d_unet_convnext_lstm optim.lr=5e-5 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=64 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=convnext_tiny.fb_in22k_ft_in1k_384 trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP9_CONVNEXT_AVG_LSTM
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=UNET3D_CONVNEXTTINY_LSTM123-ALLDATA-SEED611-LR5e-5 seed=611 exp=3d_unet_convnext_lstm optim.lr=5e-5 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=convnext_tiny.fb_in22k_ft_in1k_384 model.encoder.lstm.enable=True 'model.encoder.lstm.idxs=[1,2,3]' trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP10_R101_LR1e4
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=LR1e-4-UNET3D_R101-ALLDATA-SEED42 exp=3d_unet_smp optim.lr=1e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnet101 trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP11_MAXVIT_PICO_AVG_LSTM
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=UNET3D_MAXXVITPICO_LSTM234-ALLDATA-SEED611-LR1e-4 seed=611 exp=3d_unet_maxxvit_lstm optim.lr=1e-4 loader.train_batch_size=1 trainer.accumulate_grad_batches=4 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=maxvit_rmlp_pico_rw_256.sw_in1k model.encoder.lstm.enable=True model.encoder.lstm.num_layers=2 'model.encoder.lstm.idxs=[2,3,4]' trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP12_EFFB0_AVG_LSTM
```bash
# wrong exp_name
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=UNET3D_MAXXVITPICO_LSTM234-ALLDATA-SEED611-LR1e-4 seed=611 exp=3d_unet_efficientnet_lstm optim.lr=1e-3 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=tf_efficientnet_b0.ns_jft_in1k model.encoder.lstm.enable=True model.encoder.lstm.num_layers=2 'model.encoder.lstm.idxs=[2,3,4]' trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP13_RESNEXT50_32x4d
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_RESNEXT50-ALLDATA-SEED42-LR5e-4 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnext50_32x4d trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```


### EXP14_3DEFFB0
```bash
python3 -m yagm.run -m local=local cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_3DEFFB0-ALLDATA-SEED42-LR1e-3 exp=3d_unet_smp optim.lr=1e-3 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=tu-tf_efficientnet_b0.ns_jft_in1k trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```


### EXP15_MAXVIT_PICO_AVG_LSTM_LR5e5
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=UNET3D_MAXXVITPICO_LSTM234-ALLDATA-SEED611-LR5e-5 seed=611 exp=3d_unet_maxxvit_lstm optim.lr=5e-5 loader.train_batch_size=1 trainer.accumulate_grad_batches=4 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=maxvit_rmlp_pico_rw_256.sw_in1k model.encoder.lstm.enable=True model.encoder.lstm.num_layers=2 'model.encoder.lstm.idxs=[2,3,4]' trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=100000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null cv.train_on_all=True
```


### EXP16_X3DM_ALL_TRAIN_EXT
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM-ALLDATAEXTERNAL-SEED42 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=30000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```

<!-- ### EXP16
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=ALLGT_LR1e-4_BS8x4_CONVNEXTTINY_sigma0.2_heatmap-bce exp=2d_base_heatmap optim.lr=1e-4 loader.val_batch_size=8 loader.train_batch_size=8 trainer.accumulate_grad_batches=4 'loggers=[csv,wandb]' loader.train_num_workers=16 'data.transform.target_spacing=[32,16,16]' 'data.heatmap_stride=[16,16]' 'task.decode.heatmap_stride=[8,8]' data.agg_mode=patch data.sigma=0.2 data.sampling.rand_z_sigma_scale=1.0 'loss.enable_idxs=[2]' 'data.patch_size=[3,896,896]' data.heatmap_conf_scale_mode=null model/encoder=2d_convnext model.encoder.model_name=convnext_tiny.fb_in22k_ft_in1k_384 data.label_fname=all_gt
``` -->


### EXP17
```bash
```


### EXP18
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM_PAN-ALLDATAEXTERNAL-SEED42 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_pan model.neck.fusion_method=concat model.neck.act=relu model.neck.norm=layernorm_3d trainer.deterministic=False trainer.benchmark=False trainer.max_steps=30000 scheduler.warmup_steps=1000 trainer.val_check_interval=30000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


### EXP19
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=ALLGT_LR1e-4_BS4x8_COATLITEMEDIUM384_sigma0.2_heatmap-bce exp=2d_base_heatmap optim.lr=1e-4 loader.val_batch_size=4 loader.train_batch_size=4 trainer.accumulate_grad_batches=8 'loggers=[csv,wandb]' loader.train_num_workers=16 'data.transform.target_spacing=[32,16,16]' 'data.heatmap_stride=[16,16]' 'task.decode.heatmap_stride=[8,8]' data.agg_mode=patch data.sigma=0.2 data.sampling.rand_z_sigma_scale=1.0 'loss.enable_idxs=[2]' 'data.patch_size=[3,896,896]' data.heatmap_conf_scale_mode=null model/encoder=2d_coat data.label_fname=all_gt
```


### EXP20_2D_CONVNEXT_TINY_ALLGTV3
```bash
```


### EXP21_X3D_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM-ALLGTV3-SEED42-LR5e4 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=40000 scheduler.warmup_steps=3000 trainer.val_check_interval=20000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


### EXP22_
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_CONVNEXTTINY_AVG_LSTM-ALLGTV3-SEED611-LR5e-5 seed=611 exp=3d_unet_convnext_lstm optim.lr=5e-5 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.25 data.sampling.pre_patch.bg_from_pos_ratio=0.1 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=64 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=convnext_tiny.fb_in22k_ft_in1k_384 trainer.max_steps=40000 scheduler.warmup_steps=3000 trainer.val_check_interval=20000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=1000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


### EXP23_X3DM_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM-ALLGTV3-SEED19981106-LR5e4 seed=19981106 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=36000 scheduler.warmup_steps=3000 trainer.val_check_interval=36000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```

### EXP24_CONVNEXT_TINY_AVG_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_CONVNEXTTINY_AVG_LSTM-ALLGTV3-SEED19981022-LR5e-5 seed=19981022 exp=3d_unet_convnext_lstm optim.lr=5e-5 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=64 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.downsample.mode=avg_pool model.encoder.encoder_2d.model_name=convnext_tiny.fb_in22k_ft_in1k_384 trainer.max_steps=36000 scheduler.warmup_steps=3000 trainer.val_check_interval=36000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


### EXP25_COATLITEMEDIUM_ALLGTV3
```bash
# a30:outputs/run/multirun/05-31/22-25-45.765249_COATLITEMEDIUM384_ALLGTV3_LR1e-4_BS8x4_SEED161303
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0'  exp_name=COATLITEMEDIUM384_ALLGTV3_LR1e-4_BS8x4_SEED161303 seed=161303 exp=2d_base_heatmap optim.lr=1e-4 loader.val_batch_size=8 loader.train_batch_size=8 trainer.accumulate_grad_batches=4 'loggers=[csv,wandb]' loader.train_num_workers=32 'data.transform.target_spacing=[32,16,16]' 'data.heatmap_stride=[16,16]' 'task.decode.heatmap_stride=[8,8]' data.agg_mode=patch data.sigma=0.2 data.sampling.rand_z_sigma_scale=1.0 'loss.enable_idxs=[2]' 'data.patch_size=[3,896,896]' data.heatmap_conf_scale_mode=null model/encoder=2d_coat trainer.max_steps=20000 trainer.val_check_interval=20000 'callbacks.validation_scheduler.milestones=[999999]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null trainer.benchmark=True data.tta.enable=yx
```


### EXP26_R50_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_R50-ALLGTV3-SEED611-LR5e-4 seed=611 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=32 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnet50 trainer.max_steps=32000 scheduler.warmup_steps=3000 trainer.val_check_interval=32000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


### EXP27_2D_MAXVIT_TINY_ALLGTV3
```bash
# a30:outputs/run/multirun/05-31/21-30-22.849918_MAXVITTINY_3x896_ALLGTV3_BS8x4_LR1e4_SEED20250305
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=MAXVITTINY_3x896_ALLGTV3_BS8x4_LR1e4_SEED20250305 seed=20250305 exp=2d_base_heatmap optim.lr=1e-4 loader.val_batch_size=8 loader.train_batch_size=8 trainer.accumulate_grad_batches=4 'loggers=[csv,wandb]' loader.train_num_workers=32 'data.transform.target_spacing=[32,16,16]' 'data.heatmap_stride=[16,16]' 'task.decode.heatmap_stride=[8,8]' data.agg_mode=patch data.sampling.bg_ratio=0.05 data.sigma=0.2 data.sampling.rand_z_sigma_scale=1.0 'loss.enable_idxs=[2]' 'data.patch_size=[3,896,896]' data.heatmap_conf_scale_mode=null model/encoder=2d_maxvit model.encoder.model_name=maxvit_tiny_tf_512.in1k trainer.max_steps=20000 trainer.val_check_interval=20000 'callbacks.validation_scheduler.milestones=[999999]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null trainer.benchmark=True data.tta.enable=yx
```


### EXP28_DENSNET121_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_DENSENET121-ALLGTV3-SEED2210-LR5e-4 seed=2210 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=32 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False model.encoder.model_name=densenet121 trainer.max_steps=32000 scheduler.warmup_steps=3000 trainer.val_check_interval=32000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null
```


### EXP29_X3DL_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DL-ALLGTV3-SEED250305-LR5e4 seed=250305 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=null model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=False trainer.max_steps=36000 scheduler.warmup_steps=3000 trainer.val_check_interval=36000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx model.encoder.model_size=L model.encoder.pretrained=ckpts/x3d_l.pyth
```


### EXP30_X3DM_PAN_ALLGTV3_MIXUP
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_X3DM_PAN-MIXUP-ALLGTV3-SEED130316-LR5e4 seed=130316 exp=3d_unet_x3d optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=16 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 model/neck=3d_pan model.neck.fusion_method=concat model.neck.act=relu model.neck.norm=layernorm_3d trainer.deterministic=False trainer.benchmark=False trainer.max_steps=36000 scheduler.warmup_steps=3000 trainer.val_check_interval=36000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx data.aug.mixup_prob=0.1 data.aug.mixup_target_mode=mix data.aug.mixer_alpha=1.0
```


### EXP31_RESNEXT50_ALLGTV3
```bash
python3 -m yagm.run -m local=local data.label_fname=all_gt_v3 cv.train_on_all=True 'cv.fold_idx=0' exp_name=UNET3D_RESNEXT50-ALLGTV3-SEED6112210-LR5e-4 seed=6112210 exp=3d_unet_smp optim.lr=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=2 'loggers=[csv,wandb]' loader.train_num_workers=32 loader.val_num_workers=8 data.fast_val_workers=8 data.transform.resample_mode=trilinear 'data.transform.target_spacing=[16,16,16]' 'data.transform.heatmap_stride=[16,16,16]' 'task.decode.heatmap_stride=[16,16,16]' 'data.patch_size=[224,448,448]' data.sampling.method=pre_patch data.sampling.pre_patch.bg_ratio=0.05 data.sampling.pre_patch.bg_from_pos_ratio=0.25 data.transform.heatmap_mode='gaussian' data.sigma=0.2 data.transform.heatmap_same_std=True misc.log_model=True model.decoder.n_blocks=1 +model/neck=3d_factorized_fpn model.neck.intermediate_channels_list=32 model.neck.target_level=-2 trainer.deterministic=False trainer.benchmark=True model.encoder.model_name=resnext50_32x4d trainer.max_steps=32010 scheduler.warmup_steps=3000 trainer.val_check_interval=32000 'callbacks.validation_scheduler.milestones=[100000]' callbacks.model_checkpoint.every_n_train_steps=2000 callbacks.model_checkpoint.save_top_k=-1 callbacks.model_checkpoint.metrics=null data.tta.enable=zyx
```


# SUMMARY
## 3D
- **X3DM**:
  - `EXP16_X3DM_ALL_TRAIN_EXT` (step=30k) LB 85.8
  - PAN + MIXUP: 84.5
  - step=36k: 84.8
  - PAN: 84.1
- **X3DL** `EXP29_X3DL_ALLGTV3` (step=36k) LB 85.2
- **CONVNEXT_TINY_AVG**: `EXP22_CONVNEXT_TINY_AVG_ALLGTV3` (step=20k bug didn't save last checkpoint) LB 84.8
- **R50**: `EXP26_R50_ALLGTV3` (32k) LB 85.4
- **DENSNET121**: `EXP28_DENSENET121_ALLGTV3` (step=32k) LB 86.0
- **RESNEXT50**: `EXP31_RESNEXT50_ALLGTV3` LB 85.8

## 2D
- **MAXVIT**: `EXP27_2D_MAXVIT_TINY_ALLGTV3` step=15k, LB 85.4
- **COAT**: `EXP19_2D_COATLITEMEDIUM_ALLGTV2` step=10k, LB 84.8
- **CONVNEXT**: `EXP20_2D_CONVNEXT_TINY_ALLGTV3` step=15k, LB 83.3


## Voxel Spacing Regressor
```
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=LR1e-5_SPACING seed=42 exp=2d_base_spacing optim.lr=1e-5 loader.val_batch_size=32 loader.train_batch_size=32 trainer.accumulate_grad_batches=1 'loggers=[csv,wandb]' loader.train_num_workers=32 'data.patch_size=[512,512]'
```