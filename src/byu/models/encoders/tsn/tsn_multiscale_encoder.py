# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import logging

import torch
import torchvision
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class SlidingResample(nn.Module):

    def __init__(self, stride=1, channels=3, padding_mode="replicate"):
        super().__init__()
        # assert stride >= 1 and stride <= channels
        assert stride >= 1
        self.stride = stride
        self.channels = channels
        self.padding_mode = padding_mode

    def forward(self, x):
        """
        Args:
            x: (B, T, ...)
        """
        T = x.size(1)
        if self.channels > 2:
            # right padding
            pad = [0] * ((len(x.shape) - 2) * 2) + [
                (self.channels - 1) // 2,
                (self.channels - 1) // 2,
            ]
            x = F.pad(x, pad=pad, mode=self.padding_mode)
        xs = [
            x[:, offset : T + offset : self.stride] for offset in range(self.channels)
        ]  # List of tensor with shape (B, T2, ...)
        x = torch.stack(xs, dim=2)  # (B, T2, C, ...)
        return x


class TSN(nn.Module):
    def __init__(
        self,
        resample_stride=2,
        num_segments=8,
        base_model="resnet50",
        partial_bn=True,
        pretrain="imagenet",
        is_shift=False,
        shift_div=8,
        shift_place="blockres",
        temporal_pool=False,
        non_local=False,
    ):
        super(TSN, self).__init__()
        assert not temporal_pool  # support soon..
        self.num_segments = num_segments
        self.reshape = True
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        logger.info(
            """
Initializing TSN with base model: %s
TSN Configurations:
    num_segments:       %s
        """,
            base_model,
            self.num_segments,
        )
        self.resample = SlidingResample(stride=resample_stride, channels=3)
        self._prepare_base_model(base_model)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_base_model(self, base_model):
        logger.info("=> base model: %s", base_model)

        if "resnet" in base_model:
            self.base_model = getattr(torchvision.models, base_model)(
                True if self.pretrain == "imagenet" else False
            )
            if self.is_shift:
                logger.info("Adding temporal shift...")
                from .ops.temporal_shift import make_temporal_shift

                make_temporal_shift(
                    self.base_model,
                    self.num_segments,
                    n_div=self.shift_div,
                    place=self.shift_place,
                    temporal_pool=self.temporal_pool,
                )

            if self.non_local:
                logger.info("Adding non-local module...")
                from .ops.non_local import make_non_local

                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = "fc"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        elif base_model == "mobilenetv2":
            from archs.mobilenet_v2 import InvertedResidual, mobilenet_v2

            self.base_model = mobilenet_v2(
                True if self.pretrain == "imagenet" else False
            )

            self.base_model.last_layer_name = "classifier"
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from .ops.temporal_shift import TemporalShift

                for m in self.base_model.modules():
                    if (
                        isinstance(m, InvertedResidual)
                        and len(m.conv) == 8
                        and m.use_res_connect
                    ):
                        print("Adding temporal shift... {}".format(m.use_res_connect))
                        m.conv[0] = TemporalShift(
                            m.conv[0], n_segment=self.num_segments, n_div=self.shift_div
                        )
        elif base_model == "BNInception":
            from archs.bn_inception import bninception

            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = "fc"
            if self.is_shift:
                print("Adding temporal shift...")
                self.base_model.build_temporal_ops(
                    self.num_segments,
                    is_temporal_shift=self.shift_place,
                    shift_div=self.shift_div,
                )
        else:
            raise ValueError("Unknown base model: {}".format(base_model))

        self.register_buffer(
            "mean", torch.FloatTensor(self.input_mean).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "std", torch.FloatTensor(self.input_std).reshape(1, 3, 1, 1) * 255.0
        )
        del self.base_model.avgpool
        del self.base_model.fc

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        logger.info(
            "model.train() is called with mode=%s, partialBN=%s", mode, self._enable_pbn
        )
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            logger.info("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    @property
    def output_channels(self):
        if self.base_model_name == "resnet50":
            return [3, 64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

    def forward(self, x):
        model = self.base_model
        B, C, D, H, W = x.shape
        assert C == 1
        x = self.resample(x[:, 0])  # (B, T2, C, H, W)
        # BTCHW -> (BT)CHW
        x = x.flatten(0, 1)
        x = (x - self.mean) / self.std

        features = [x]
        if self.base_model_name == "resnet50":
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            features.append(x)  # /2
            x = model.maxpool(x)
            x = model.layer1(x)
            features.append(x)  # /4
            x = model.layer2(x)
            features.append(x)  # /8
            x = model.layer3(x)
            features.append(x)  # /16
            x = model.layer4(x)
            features.append(x)  # /32
        features = [
            feat.reshape(B, -1, *feat.shape[-3:]).permute(0, 2, 3, 4, 1)
            for feat in features
        ]  # (BD)CHW -> BDCHW -> BCHWD

        # print([e.shape for e in features])
        return features


class TSNEncoder(TSN):
    def __init__(self, global_cfg):
        encoder_kwargs = OmegaConf.to_object(global_cfg.model.encoder)
        encoder_kwargs.pop("_target_")
        pretrain_path = encoder_kwargs.pop("pretrained")
        super().__init__(**encoder_kwargs)
        # load pretrain weights
        if pretrain_path is not None:
            logger.info("Loading state dict from %s", pretrain_path)
            sd = torch.load(pretrain_path)["state_dict"]
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            # print(sd.keys())
            model_keys = set(self.state_dict().keys())
            sd_keys = set(sd.keys())
            unmatched_keys = list(
                model_keys.union(sd_keys).difference(model_keys.intersection(sd_keys))
            )
            logger.info("Unmatched keys: %s", unmatched_keys)
            self.load_state_dict(sd, strict=False)
