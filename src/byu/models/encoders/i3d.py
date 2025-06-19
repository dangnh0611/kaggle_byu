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


class I3DEncoder(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        # ========== LOAD ENCODER =============
        slowfast_cfg = get_slowfast_default_cfg()
        slowfast_cfg.merge_from_file(
            f"third_party/slowfast/configs/Kinetics/c2/I3D_NLN_8x8_R50.yaml"
        )
        slowfast_cfg.DATA.INPUT_CHANNEL_NUM = [3]
        slowfast_cfg.MODEL.NUM_CLASSES = 0
        # slowfast_cfg.MODEL.ARCH = "i3d"
        # slowfast_cfg.MODEL.MODEL_NAME = "ResNet"
        # if cfg.encoder.pretrained:
        #     slowfast_cfg.TRAIN.CHECKPOINT_FILE_PATH = cfg.encoder.pretrained
        slowfast_cfg.NUM_GPUS = 0
        # print(slowfast_cfg)
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

        # M: (24, 24, 48, 96, 192)
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
        B, C, D, H, W = x.shape
        # mean/std norm, same as SlowFast
        # x = (x - 0.45) / 0.225
        x = (x - 114.75) / 57.375
        x = x.expand(-1, 3, -1, -1, -1)
        # BCDHW (BCTHW for video model)
        features = self.encoder([x])
        features = [e.permute(0, 1, 3, 4, 2) for e in features]  # BCDHW -> BCHWD
        return features
