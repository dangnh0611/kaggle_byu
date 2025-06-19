# @TODO: add norm before Unet decoder ?

import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm.layers import SelectAdaptivePool2d, get_act_layer

logger = logging.getLogger(__name__)


class MlpHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: Optional[int] = None,
        drop_rate: float = 0.0,
        act_layer: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        act_layer = get_act_layer(act_layer)

        if hidden_size:
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(in_features, hidden_size)),
                        ("act", act_layer()),
                    ]
                )
            )
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x):
        x = self.pre_logits(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


def get_feature_channels(model):
    if hasattr(model, "feature_info"):
        print("Feature info:", model.feature_info)
        if hasattr(model.feature_info, "channels"):
            feature_channels = model.feature_info.channels()
        else:
            feature_channels = [
                stage_info["num_chs"] for stage_info in model.feature_info
            ]
    elif hasattr(model, "embed_dims"):
        feature_channels = model.embed_dims
    else:
        raise AssertionError("Unknown model's intermediate output channels!")
    return feature_channels


class Spacing2dModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._model_name = cfg.encoder.model_name
        self.encoder = timm.create_model(**cfg.encoder)
        from timm.data import resolve_data_config

        data_config = resolve_data_config({}, model=self.encoder)
        logger.info("Encoder data config: %s", data_config)

        if getattr(cfg, "normalize", "model") == "model":
            logger.info("Normalize by model")
            mean = torch.tensor(data_config["mean"], dtype=torch.float32) * 255
            std = torch.tensor(data_config["std"], dtype=torch.float32) * 255
            self.register_buffer("mean", torch.FloatTensor(mean).reshape(1, 3, 1, 1))
            self.register_buffer("std", torch.FloatTensor(std).reshape(1, 3, 1, 1))
        else:
            logger.info("Normalize by 127.5")
            self.mean = 127.5
            self.std = 127.5

        if hasattr(self.encoder, "reset_classifier"):
            self.encoder.reset_classifier(num_classes=0)
        if hasattr(self.encoder, "prune_intermediate_layers"):
            self.encoder.prune_intermediate_layers(
                prune_norm=True,
                prune_head=True,
            )
        enc_channels = get_feature_channels(self.encoder)
        print(f"Multiscale Feature Channels: {enc_channels}")

        # IMAGE HEAD
        self.img_global_pool = SelectAdaptivePool2d(
            pool_type=cfg.head.pool_type, flatten=True
        )
        self.img_dropout = nn.Dropout(p=cfg.head.drop_rate)
        self.regress = MlpHead(
            in_features=enc_channels[-1],
            num_classes=1,
            hidden_size=cfg.head.hidden_size,
            drop_rate=cfg.head.hidden_drop_rate,
            act_layer=cfg.head.hidden_act,
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.encoder.forward_features(x)
        x = self.img_global_pool(x)
        x = self.img_dropout(x)
        logit = self.regress(x)
        prob = F.sigmoid(logit)
        return prob
