# @TODO: add norm before Unet decoder ?

import logging
from typing import Callable, Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from yagm.layers.activation import get_act
from yagm.layers.helper import get_timm_feature_channels, reset_timm_head
from yagm.layers.norms import get_norm

logger = logging.getLogger(__name__)


# init so that it's equal to avg pooling
def init_conv3d_as_avgpool(conv: nn.Conv3d):
    # Kernel volume = D × H × W
    k = conv.kernel_size
    kernel_volume = k[0] * k[1] * k[2]
    weight_value = 1.0 / kernel_volume

    # Set weight: shape (out_channels, in_channels, D, H, W)
    with torch.no_grad():
        conv.weight.fill_(0.0)
        for out_c in range(conv.out_channels):
            for in_c in range(conv.in_channels):
                conv.weight[out_c, in_c].fill_(weight_value)
        if conv.bias is not None:
            conv.bias.zero_()


class LSTMBlock(nn.Module):

    def __init__(self, in_chs, bidirectional=False, num_layers=1):
        """LSTM to encode a sequence of 2D images along time/depth dimension.
        Expect input of shape (B, T, C, H, W) or (B, D, C, H, W)
        Args:
            in_chs: number of input channels
            bidirectional: use bidirectional LSTM or not
            num_layers: number of LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(
            in_chs,
            in_chs if not bidirectional else in_chs // 2,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

    def forward(self, x):
        # @TODO - optimize?
        B, T, C, H, W = x.shape
        # BTCHW -> BTC(HW) -> B(HW)TC -> (BHW)TC
        x = x.flatten(-2, -1).permute(0, 3, 1, 2).flatten(0, 1)
        x = self.lstm(x)[0]  # (BHW)TC
        x = x.view(-1, H, W, T, C).permute(0, 3, 4, 1, 2)  # BHWTC -> BTCHW
        return x


class SlidingResample(nn.Module):

    def __init__(self, stride=1, channels=3, padding_mode="replicate"):
        super().__init__()
        assert stride >= 1 and stride <= channels
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


from typing import List, Tuple

from timm.models._features import feature_take_indices


def _convnext_forward_intermediates(
    self,
    x: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
    norm: bool = False,
    stop_early: bool = False,
    output_fmt: str = "NCHW",
    intermediates_only: bool = False,
    batch_size=None,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward features that returns intermediates.

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, all if None, select matching indices if sequence
        norm: Apply norm layer to compatible intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
    Returns:

    """
    assert output_fmt in ("NCHW",), "Output shape must be NCHW."
    intermediates = []
    take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

    # forward pass
    feat_idx = 0  # stem is index 0
    x = self.stem(x)
    if feat_idx in take_indices:
        intermediates.append(x)

    if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
        stages = self.stages
    else:
        stages = self.stages[:max_index]

    down_idx = -1
    for stage in stages:
        feat_idx += 1
        x = stage(x)
        if feat_idx in take_indices:
            # NOTE not bothering to apply norm_pre when norm=True as almost no models have it enabled
            down_idx += 1
            down_module = self.down_modules[down_idx]
            if not isinstance(down_module, nn.Identity):
                x = x.view(batch_size, -1, *x.shape[1:]).permute(
                    0, 2, 1, 3, 4
                )  # (BT)CHW -> BTCHW -> BCTHW
                x = down_module(x)
                x = x.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW
                x = x.reshape(-1, *x.shape[2:])  # BTCHW -> (BT)CHW
            intermediates.append(x)

    if intermediates_only:
        return intermediates

    x = self.norm_pre(x)

    return x, intermediates


def _maxxvit_forward_intermediates(
    self,
    x: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
    norm: bool = False,
    stop_early: bool = False,
    output_fmt: str = "NCHW",
    intermediates_only: bool = False,
    batch_size=None,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward features that returns intermediates.

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, all if None, select matching indices if sequence
        norm: Apply norm layer to compatible intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
    Returns:

    """
    assert output_fmt in ("NCHW",), "Output shape must be NCHW."
    intermediates = []
    take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

    # forward pass
    feat_idx = 0  # stem is index 0
    x = self.stem(x)
    if feat_idx in take_indices:
        intermediates.append(x)

    last_idx = len(self.stages)
    if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
        stages = self.stages
    else:
        stages = self.stages[:max_index]

    down_idx = -1
    for stage in stages:
        feat_idx += 1
        x = stage(x)
        if feat_idx in take_indices:
            down_idx += 1
            down_module = self.down_modules[down_idx]
            if not isinstance(down_module, nn.Identity):
                x = x.view(batch_size, -1, *x.shape[1:]).permute(
                    0, 2, 1, 3, 4
                )  # (BT)CHW -> BTCHW -> BCTHW
                x = down_module(x)
                x = x.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW
                x = x.reshape(-1, *x.shape[2:])  # BTCHW -> (BT)CHW
            intermediates.append(x)

    if intermediates_only:
        return intermediates

    x = self.norm(x)

    return x, intermediates


def _eff_forward_intermediates(
    self,
    x: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
    norm: bool = False,
    stop_early: bool = False,
    output_fmt: str = "NCHW",
    intermediates_only: bool = False,
    extra_blocks: bool = False,
    batch_size=None,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Forward features that returns intermediates.

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, all if None, select matching indices if sequence
        norm: Apply norm layer to compatible intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
        extra_blocks: Include outputs of all blocks and head conv in output, does not align with feature_info
    Returns:

    """
    assert output_fmt in ("NCHW",), "Output shape must be NCHW."
    intermediates = []
    if extra_blocks:
        take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
    else:
        take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
        take_indices = [self.stage_ends[i] for i in take_indices]
        max_index = self.stage_ends[max_index]
    # forward pass
    feat_idx = 0  # stem is index 0
    x = self.conv_stem(x)
    x = self.bn1(x)
    if feat_idx in take_indices:
        intermediates.append(x)

    if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
        blocks = self.blocks
    else:
        blocks = self.blocks[:max_index]

    down_idx = -1
    for blk in blocks:
        feat_idx += 1
        x = blk(x)
        if feat_idx in take_indices:
            down_idx += 1
            down_module = self.down_modules[down_idx]
            if feat_idx != 1 and not isinstance(down_module, nn.Identity):
                x = x.view(batch_size, -1, *x.shape[1:]).permute(
                    0, 2, 1, 3, 4
                )  # (BT)CHW -> BTCHW -> BCTHW
                x = down_module(x)
                x = x.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW
                x = x.reshape(-1, *x.shape[2:])  # BTCHW -> (BT)CHW
            intermediates.append(x)

    if intermediates_only:
        return intermediates

    if feat_idx == self.stage_ends[-1]:
        x = self.conv_head(x)
        x = self.bn2(x)

    return x, intermediates


class CnnLstmEncoder(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model.encoder

        self.resample = SlidingResample(
            stride=cfg.resample.stride, channels=cfg.resample.channels
        )

        encoder_2d_kwargs = OmegaConf.to_object(cfg.encoder_2d)
        self._feature_indices = encoder_2d_kwargs.pop("feature_indices")
        self.encoder_2d = timm.create_model(**encoder_2d_kwargs)
        from timm.data import resolve_data_config

        data_config = resolve_data_config({}, model=self.encoder_2d)
        logger.info("Encoder data config: %s", data_config)

        self._normalize = False
        if cfg.normalize is None:
            pass
        elif cfg.normalize == "model":
            self._normalize = True
            logger.info("Normalize by model's associated mean/std")
            mean = torch.tensor(data_config["mean"], dtype=torch.float32)
            std = torch.tensor(data_config["std"], dtype=torch.float32)
            import math

            _expand = math.ceil(cfg.resample.channels / 3)
            self.register_buffer(
                "mean",
                torch.FloatTensor(mean)
                .reshape(1, 3, 1, 1)
                .repeat(1, _expand, 1, 1)[:, : cfg.resample.channels]
                * 255.0,
            )
            self.register_buffer(
                "std",
                torch.FloatTensor(std)
                .reshape(1, 3, 1, 1)
                .repeat(1, _expand, 1, 1)[:, : cfg.resample.channels]
                * 255.0,
            )
        else:
            raise ValueError

        reset_timm_head(self.encoder_2d)
        enc_channels = get_timm_feature_channels(self.encoder_2d)
        logger.info("Encoder channels: %s", enc_channels)
        enc_channels = [enc_channels[idx] for idx in self._feature_indices]
        logger.info(f"Multiscale Feature Channels: {enc_channels}")

        self._model_name = cfg.encoder_2d.model_name

        self._use_lstm = False
        if cfg.lstm.enable:
            self._use_lstm = True
            self.lstm_layer_idxs = list(cfg.lstm.idxs)
            # to handle negative index
            lstms = [nn.Identity() for _ in range(len(self._feature_indices))]
            for idx in self.lstm_layer_idxs:
                lstms[idx] = LSTMBlock(
                    enc_channels[idx],
                    bidirectional=cfg.lstm.bi,
                    num_layers=cfg.lstm.num_layers,
                )
            self.lstms = nn.ModuleList(lstms)

        # add intermediate depth mixer or depth downsampling
        downsample_mode = cfg.downsample.mode
        down_modules = []
        assert len(cfg.downsample.strides) == len(enc_channels)

        for channel, stride in zip(enc_channels, cfg.downsample.strides):
            if downsample_mode is None:
                down_module = nn.Identity()
            elif downsample_mode == "conv333":
                down_module = nn.Conv3d(
                    channel,
                    channel,
                    (3, 3, 3),
                    stride=(stride, 1, 1),
                    padding=(1, 1, 1),
                    groups=1,
                    bias=True,
                )
                init_conv3d_as_avgpool(
                    down_module
                )  # can be harmfull when combined with act+norm
            elif downsample_mode == "can333":
                # Conv + Act + Norm with kernel size (3,3,3)
                down_module = nn.Sequential(
                    nn.Conv3d(
                        channel,
                        channel,
                        (3, 3, 3),
                        stride=(stride, 1, 1),
                        padding=(1, 1, 1),
                        groups=1,
                        bias=True,
                    ),
                    nn.GELU(),
                    get_norm("layernorm_3d", channel_first=True)(channel),
                )
            elif downsample_mode == "2conv":
                down_module = nn.Sequential(
                    nn.Conv3d(
                        channel,
                        channel,
                        kernel_size=(3, 3, 3),
                        stride=(stride, 1, 1),
                        padding=(1, 1, 1),
                    ),
                    get_act("gelu")(),
                    get_norm("layernorm_3d", channel_first=True)(channel),
                    nn.Conv3d(
                        channel,
                        channel,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                    ),
                )
            elif downsample_mode == "avg_pool":
                down_module = nn.AvgPool3d((stride, 1, 1))
            elif downsample_mode == "max_pool":
                down_module = nn.MaxPool3d((stride, 1, 1))
            else:
                raise NotImplementedError
            down_modules.append(down_module)
            self.encoder_2d.down_modules = nn.ModuleList(down_modules)

            import types

            if "convnext" in cfg.encoder_2d.model_name:
                self.encoder_2d.forward_intermediates = types.MethodType(
                    _convnext_forward_intermediates, self.encoder_2d
                )
            elif "maxvit" in cfg.encoder_2d.model_name:
                self.encoder_2d.forward_intermediates = types.MethodType(
                    _maxxvit_forward_intermediates, self.encoder_2d
                )
            elif "efficientnet" in cfg.encoder_2d.model_name:
                self.encoder_2d.forward_intermediates = types.MethodType(
                    _eff_forward_intermediates, self.encoder_2d
                )
            else:
                raise ValueError

        self._output_channels = [1] + enc_channels
        logger.info("Encoder output channels:%s", self._output_channels)

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert C == 1
        x = self.resample(x[:, 0])  # (B, T2, C, H, W)
        T = x.shape[1]
        # BTCHW -> (BT)CHW
        x = x.flatten(0, 1)

        if self._normalize:
            x = (x - self.mean) / self.std

        if "coat_" in self._model_name:
            enc_features = self.encoder_2d.forward_features(x)
            enc_features = list(enc_features.values())
        elif "convnext" in self._model_name:
            # convnext: stem has feature index 0
            # after stage 1, resolution is not reduce
            # so we use output of stage 1 as stride 4 features (first downscale feature)
            enc_features = self.encoder_2d.forward_intermediates(
                x,
                indices=[1 + e for e in self._feature_indices],
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
                batch_size=B,
            )
        elif (
            "maxvit" in self._model_name
            or "coatnet" in self._model_name
            or "swinv2" in self._model_name
        ):
            enc_features = self.encoder_2d.forward_intermediates(
                x,
                indices=self._feature_indices,
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
                batch_size=B,
            )
        elif "efficientnet" in self._model_name:
            enc_features = self.encoder_2d.forward_intermediates(
                x,
                indices=self._feature_indices,
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
                extra_blocks=False,
                batch_size=B,
            )
        else:
            raise AssertionError

        # (BT)CHW -> BTCHW
        enc_features = [feat.view(B, -1, *feat.shape[1:]) for feat in enc_features]

        # for i, feat in enumerate(enc_features):
        #     print(i, feat.shape)

        if self._use_lstm and T > 1:
            for idx in self.lstm_layer_idxs:
                enc_features[idx] = self.lstms[idx](enc_features[idx])  # BTCHW
        enc_features = [
            feat.permute(0, 2, 3, 4, 1) for feat in enc_features
        ]  # BTCHW -> BCHWT

        # print([e.shape for e in enc_features])

        return [None] + enc_features


if __name__ == "__main__":
    yaml_str = """

model:
    encoder:
        _target_: null
        
        resample:
            stride: 2
            channels: 3
            
        normalize: model
            
        # encoder_2d:        
        #     model_name: convnext_nano.r384_ad_in12k
        #     pretrained: True
        #     feature_indices: [0, 1, 2, 3]
        #     features_only: False
        #     in_chans: ${model.encoder.resample.channels}
        #     drop_rate: 0.0
        #     drop_path_rate: 0.0

        # encoder_2d:        
        #     model_name: maxvit_rmlp_pico_rw_256.sw_in1k  # maxvit_tiny_tf_512.in1k
        #     feature_indices: [0, 1, 2, 3, 4]
        #     pretrained: True
        #     img_size: [448,448]
        #     in_chans: ${model.encoder.resample.channels}
        #     drop_rate: 0.0
        #     drop_path_rate: 0.0

        encoder_2d:
            model_name: tf_efficientnet_b5.ns_jft_in1k
            pretrained: True
            feature_indices: [0, 1, 2, 3, 4]
            features_only: False
            in_chans: ${model.encoder.resample.channels}
            drop_rate: 0.0
            drop_path_rate: 0.0

        downsample:
            mode: avg_pool
            strides: [2, 2, 2, 2, 2]

        lstm:
            idxs: [2, 3]
            enable: False
            bi: True
            num_layers: 1

        decoder:
            dropout: 0.0
            out_channels: 3
"""
    logging.basicConfig(level=logging.INFO)
    global_cfg = OmegaConf.create(yaml_str)
    print(global_cfg)
    model = CnnLstmEncoder(global_cfg)
    model.cuda().eval()
    print(model)

    with torch.inference_mode():
        inp = torch.rand((1, 1, 64, 448, 448)).cuda()
        out = model(inp)
        print("Input:", inp.shape)
        # print("Ouput:", {k: v.shape for k, v in out.items()})
        print([v.shape if isinstance(v, torch.Tensor) else v for v in out])
