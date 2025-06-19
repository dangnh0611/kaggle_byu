import gc
import logging

import hydra
import segmentation_models_pytorch_3d as smp
import slowfast.utils.checkpoint as cu
import torch
import torch.nn as nn
import torch.nn.functional as F
from hiera.hiera import Hiera
from omegaconf import OmegaConf
from segmentation_models_pytorch_3d.base import SegmentationHead
from segmentation_models_pytorch_3d.decoders.unet.decoder import MultiscaleUnetDecoder
from segmentation_models_pytorch_3d.encoders import get_encoder as get_encoder_smp

from byu.models.encoders import mvitv2, x3d

logger = logging.getLogger(__name__)


class Hiera3dEncoder(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        encoder_kwargs = OmegaConf.to_object(global_cfg.model.encoder)
        _ = encoder_kwargs.pop("_target_")

        encoder = Hiera(**encoder_kwargs)
        self.in_chans = encoder_kwargs["in_chans"]

        self.encoder = encoder

        # _expand = 1
        # self.register_buffer(
        #     "mean",
        #     torch.FloatTensor([0.485, 0.456, 0.406])
        #     .reshape(1, 3, 1, 1, 1)
        #     .repeat(1, _expand, 1, 1, 1)
        #     * 255.0,
        # )
        # self.register_buffer(
        #     "std",
        #     torch.FloatTensor([0.229, 0.224, 0.225])
        #     .reshape(1, 3, 1, 1, 1)
        #     .repeat(1, _expand, 1, 1, 1)
        #     * 255.0,
        # )
        self.mean, self.std = 0.45, 0.225

    @property
    def output_channels(self):
        return [self.in_chans] + list(self.encoder.output_channels)

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)  # BCDHW -> BCHWD
        # x = x.expand(-1, 3, -1, -1, -1)  # 1 to 3 channels
        x = (x - self.mean) / self.std
        _, features = self.encoder(x, return_intermediates=True)
        features = [e.permute(0, 4, 2, 3, 1) for e in features]  # BDHWC -> BCHWD
        return [None] + features


if __name__ == "__main__":
    yaml_str = """

model:
    encoder:
        _target_: byu.models.encoders.hiera_3d.Hiera3dEncoder
        num_classes: 1
        in_chans: 3
        input_size: [128,448,448]
        embed_dim: 64
        num_heads: 1
        stages: [1, 2, 7, 2]  # T
        q_stride: [2, 2, 2]  # ori (1,2,2)
        mask_unit_size: [1, 8, 8]
        patch_kernel: [3, 7, 7]
        patch_stride: [4, 4, 4] # ori (2, 4, 4)
        patch_padding: [1, 3, 3]
        sep_pos_embed: True
"""
    device = torch.device("cuda")
    logging.basicConfig(level=logging.INFO)
    global_cfg = OmegaConf.create(yaml_str)
    print(global_cfg)
    model = Hiera3dEncoder(global_cfg).to(device)
    print(model)
    print("OUTPUT CHANNELS:", model.output_channels)

    # with torch.inference_mode():

    # with torch.inference_mode():
    inp = torch.rand((1, 1, 128, 448, 448)).to(device)
    for i in range(50):
        out = model(inp)
        print("Input:", inp.shape)
        # print("Ouput:", {k: v.shape for k, v in out.items()})
        print("Output: ", [v.shape if isinstance(v, torch.Tensor) else v for v in out])
