import gc
import logging

import segmentation_models_pytorch_3d as smp
import slowfast.utils.checkpoint as cu
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from segmentation_models_pytorch_3d.base import SegmentationHead
from segmentation_models_pytorch_3d.decoders.unet.decoder import MultiscaleUnetDecoder
from segmentation_models_pytorch_3d.encoders import get_encoder as get_encoder_smp
from slowfast.config.defaults import get_cfg as get_slowfast_default_cfg
from slowfast.models import build_model as build_slowfast_model

logger = logging.getLogger(__name__)

MVITV2_CONFIG_PATHS = {
    "S": "third_party/slowfast/configs/Kinetics/MVITv2_S_16x4.yaml",
    "B": "third_party/slowfast/configs/Kinetics/MVITv2_B_32x3.yaml",
}


class MVITv2Encoder(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        slowfast_cfg = get_slowfast_default_cfg()
        # @TODO - check HARD CODE in def forward if change the version
        slowfast_cfg.merge_from_file(MVITV2_CONFIG_PATHS[cfg.encoder.model_size])
        # print(slowfast_cfg)
        slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [3]
        slowfast_cfg.DATA.NUM_FRAMES = global_cfg.data.patch_size[0]
        assert global_cfg.data.patch_size[1] == global_cfg.data.patch_size[2]
        slowfast_cfg.DATA.TRAIN_CROP_SIZE = global_cfg.data.patch_size[1]
        slowfast_cfg.DATA.TEST_CROP_SIZE = slowfast_cfg.DATA.TRAIN_CROP_SIZE
        slowfast_cfg.MODEL.NUM_CLASSES = 0
        slowfast_cfg.MODEL.ARCH = "mvitv2encoder"
        slowfast_cfg.MODEL.MODEL_NAME = "MViTEncoder"
        slowfast_cfg.MVIT.CLS_EMBED_ON = False
        # slowfast_cfg.MVIT.PATCH_STRIDE = [2, 4, 4]
        slowfast_cfg.NUM_GPUS = 0
        encoder = build_slowfast_model(slowfast_cfg)
        if cfg.encoder.pretrained:
            logger.info("Load from checkpoint %s", cfg.encoder.pretrained)
            cu.load_checkpoint(
                cfg.encoder.pretrained,
                encoder,
                data_parallel=False,
                optimizer=None,
                scaler=None,
                inflation=slowfast_cfg.TRAIN.CHECKPOINT_INFLATE,
                convert_from_caffe2=slowfast_cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
                epoch_reset=slowfast_cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                clear_name_pattern=slowfast_cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
                image_init=slowfast_cfg.TRAIN.CHECKPOINT_IN_INIT,
            )
        self._output_channels = encoder.output_channels
        # @TODO - prune un-used modules
        # encoder.head = nn.Identity()
        gc.collect()
        self.encoder = encoder

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: BCDHW
        Returns:
            Multiscale features of shape BCHWD
        """
        # mean/std norm, same as SlowFast
        # x = (x - 0.45) / 0.225
        x = x.expand(-1, 3, -1, -1, -1)
        features = self.encoder([x])
        # @TODO - improve
        features = [e.permute(0, 1, 3, 4, 2) for e in features]  # BCDHW -> BCHWD
        return features
