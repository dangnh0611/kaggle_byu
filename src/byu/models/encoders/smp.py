import gc
import logging

import hydra
import segmentation_models_pytorch_3d as smp
import slowfast.utils.checkpoint as cu
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from segmentation_models_pytorch_3d.base import SegmentationHead
from segmentation_models_pytorch_3d.decoders.unet.decoder import MultiscaleUnetDecoder
from segmentation_models_pytorch_3d.encoders import get_encoder as get_encoder_smp

from byu.models.encoders import mvitv2, x3d

logger = logging.getLogger(__name__)


class SMPEncoder(nn.Module):
    def __init__(self, global_cfg):
        super().__init__()
        encoder_cfg = OmegaConf.to_object(global_cfg.model.encoder)
        model_name = encoder_cfg.pop("model_name")
        _ = encoder_cfg.pop("_target_")
        encoder = get_encoder_smp(name=model_name, **encoder_cfg)
        self.encoder = encoder
        print("ENCODER:", self.encoder.out_channels)

        _expand = 1
        self.register_buffer(
            "mean",
            torch.FloatTensor([0.485, 0.456, 0.406])
            .reshape(1, 3, 1, 1, 1)
            .repeat(1, _expand, 1, 1, 1)
            * 255.0,
        )
        self.register_buffer(
            "std",
            torch.FloatTensor([0.229, 0.224, 0.225])
            .reshape(1, 3, 1, 1, 1)
            .repeat(1, _expand, 1, 1, 1)
            * 255.0,
        )

    @property
    def output_channels(self):
        return list(self.encoder.out_channels)

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)  # BCDHW -> BCHWD
        x = x.expand(-1, 3, -1, -1, -1)  # 1 to 3 channels
        x = (x - self.mean) / self.std
        features = self.encoder(x)
        return features


if __name__ == "__main__":
    yaml_str = """

model:
    encoder:
        _target_: byu.models.encoders.smp.SMPEncoder
        model_name: densenet161
        weights: imagenet
        out_indices: [0,1,2,3]
"""
    device = torch.device("cpu")
    logging.basicConfig(level=logging.INFO)
    global_cfg = OmegaConf.create(yaml_str)
    print(global_cfg)
    model = SMPEncoder(global_cfg).to(device)
    print(model)

    # with torch.inference_mode():
    inp = torch.rand((1, 1, 224, 448, 448)).to(device)
    for i in range(1):
        out = model(inp)
        print("Input:", inp.shape)
        # print("Ouput:", {k: v.shape for k, v in out.items()})
        print("Output: ", [v.shape if isinstance(v, torch.Tensor) else v for v in out])
