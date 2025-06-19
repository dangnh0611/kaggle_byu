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
from timm.data import resolve_data_config
from timm.layers import LayerNorm2d, SelectAdaptivePool2d, get_act_layer, get_norm_layer
from yagm.layers.helper import get_timm_feature_channels, reset_timm_head
from yagm.layers.neck import FactorizedFPN2d

logger = logging.getLogger(__name__)


from typing import List, Tuple

from timm.models._features import feature_take_indices


############## EFFICIENTVIT_MIT PATCHING #############
def _efficientvit_forward_intermediates(
    self,
    x: torch.Tensor,
    indices: Optional[Union[int, List[int]]] = None,
    norm: bool = False,
    stop_early: bool = False,
    output_fmt: str = "NCHW",
    intermediates_only: bool = False,
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
    take_indices, max_index = feature_take_indices(len(self.stages), indices)

    # forward pass
    x = self.stem(x)

    if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
        stages = self.stages
    else:
        stages = self.stages[: max_index + 1]

    for feat_idx, stage in enumerate(stages):
        x = stage(x)
        if feat_idx in take_indices:
            intermediates.append(x)

    if intermediates_only:
        return intermediates

    return x, intermediates


class PixelShuffleICNR(nn.Sequential):

    def __init__(self, ni, nf=None, scale=2, blur=True):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [
            nn.Conv2d(ni, nf * (scale**2), 1),
            LayerNorm2d(nf * (scale**2)),
            nn.GELU(),
            nn.PixelShuffle(scale),
        ]
        layers[0].weight.data.copy_(self.icnr_init(layers[0].weight.data))
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

    def icnr_init(self, x, scale=2, init=nn.init.kaiming_normal_):
        "ICNR init of `x`, with `scale` and `init` function"
        ni, nf, h, w = x.shape
        ni2 = int(ni / (scale**2))
        k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale**2)
        return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class UnetBlock(nn.Module):

    def __init__(
        self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False, **kwargs
    ):
        super().__init__()
        self.shuf = PixelShuffleICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        self.bn = LayerNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = nn.Sequential(nn.Conv2d(ni, nf, 3, padding=1), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.GELU())
        self.relu = nn.GELU()

    def forward(self, up_in: torch.Tensor, left_in: torch.Tensor) -> torch.Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class DecoupledRegressionHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        cls_out_dim: int,
        reg_out_dim: int,
        hidden_size: Optional[int] = None,
        pool_type: str = "avg",
        first_drop_rate: float = 0.0,
        reg_drop_rate: float = 0.0,
        cls_drop_rate: float = 0.0,
        norm_layer: Union[str, Callable] = "layernorm2d",
        act_layer: Union[str, Callable] = "tanh",
    ):
        """
        Ref: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py
        Maxxvit MLP head: `Pool -> Norm -> Drop -> FC1 -> ACT1 -> (FC2 + FC3)`.
        This is different from usual head `Norm -> Pool -> FC`
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            hidden_size: The hidden size of the MLP (pre-logits FC layer) if not None.
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
            norm_layer: Normalization layer type.
            act_layer: MLP activation layer type (only used if hidden_size is not None).
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Linear

        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("drop", nn.Dropout(first_drop_rate)),
                        ("fc", linear_layer(in_features, hidden_size)),
                        ("act", act_layer()),
                    ]
                )
            )
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        if reg_drop_rate > 0.0:
            logger.warning(
                "Dropout was set to %f > 0 in regression head!", reg_drop_rate
            )
        self.reg_drop = nn.Dropout(reg_drop_rate)
        self.cls_drop = nn.Dropout(cls_drop_rate)
        self.reg_fc = linear_layer(self.num_features, reg_out_dim)
        self.cls_fc = linear_layer(self.num_features, cls_out_dim)

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        raise NotImplementedError

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        reg_logits = self.reg_fc(self.reg_drop(x))
        cls_logits = self.cls_fc(self.cls_drop(x))
        return cls_logits, reg_logits


class DecoupledRegressionHeadV2(nn.Module):

    def __init__(
        self,
        in_features: int,
        num_kpts: int,
        feature_h: int,
        feature_w: int,
        hidden_size_per_kpt: int = 1,
        hidden_size: Optional[int] = None,
        first_drop_rate: float = 0.0,
        reg_drop_rate: float = 0.0,
        cls_drop_rate: float = 0.0,
        norm_layer: Union[str, Callable] = "layernorm2d",
        act_layer: Union[str, Callable] = "tanh",
    ):
        """
        No pooling, flatten along spatial dimensions
        """
        super().__init__()
        self.K = num_kpts
        self.D = hidden_size_per_kpt
        self.H = feature_h
        self.W = feature_w
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)

        self.norm = norm_layer(in_features)
        self.proj_conv = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.K * self.D,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_act = act_layer()

        if hidden_size:
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("drop", nn.Dropout(first_drop_rate)),
                        ("fc", nn.Linear(self.D * self.H * self.W, hidden_size)),
                        ("act", act_layer()),
                    ]
                )
            )
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        if reg_drop_rate > 0.0:
            logger.warning(
                "Dropout was set to %f > 0 in regression head!", reg_drop_rate
            )
        self.reg_drop = nn.Dropout(reg_drop_rate)
        self.cls_drop = nn.Dropout(cls_drop_rate)
        self.reg_fc = nn.Linear(self.num_features, 2)
        self.cls_fc = nn.Linear(self.num_features, 1)

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        raise NotImplementedError

    def forward(self, x):
        x = self.norm(x)
        x = self.proj_act(self.proj_conv(x))
        B, KD, H, W = x.shape
        x = x.reshape(B, self.K, self.D, H, W).reshape(B, self.K, self.D * H * W)
        x = self.pre_logits(x)
        reg_logits = self.reg_fc(self.reg_drop(x)).reshape(
            B, -1
        )  # (B, K, 2) -> (B, 2K)
        cls_logits = self.cls_fc(self.cls_drop(x)).reshape(B, -1)
        return cls_logits, reg_logits


class DSNT2d(nn.Module):
    """
    Differentiable spatial to numerical transform (DSNT)
    Ref: https://arxiv.org/abs/1801.07372
    """

    def __init__(self, height, width, act="sigmoid"):
        super().__init__()
        self._act = act
        # xy
        self.grid = self.create_normalized_meshgrid(height, width)
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "identity":
            self.act = nn.Identity()
        elif act == "softmax":
            pass
        else:
            raise ValueError

    def create_normalized_meshgrid(self, height, width):
        """
        Return meshgrid in range [0, 1], shape (H, W, 2)
        """
        xs = torch.linspace(0.5 / width, 1 - 0.5 / width, width)
        ys = torch.linspace(0.5 / height, 1 - 0.5 / height, height)
        grid = torch.stack(torch.meshgrid([xs, ys], indexing="xy"), dim=-1)
        return grid

    def forward(self, heatmap):
        """
        Args:
            heatmap: heatmap raw logits of shape (N, C, H, W)
        Returns:
            Normalized coordinates (x,y) in range [0,1] in coordinate space (not pixel space)
            shape (N, C, 2)
        """
        B, C, H, W = heatmap.shape
        if self._act != "softmax":
            heatmap = self.act(heatmap)
            prob = heatmap / heatmap.sum(dim=(2, 3), keepdim=True)
        else:
            heatmap = torch.flatten(heatmap, -2, -1)
            prob = F.softmax(heatmap, dim=-1).reshape(B, C, H, W)
        # (1,1,H,W,2) * (N,C,H,W,1) -> (N,C,H,W,2) -> (N,C,2)
        xy = torch.sum(self.grid[None, None].to(prob) * prob[..., None], dim=(2, 3))
        return xy


class Unet2dModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model
        self._model_name = cfg.encoder.model_name
        self._num_kpts = 1

        self.encoder = timm.create_model(**cfg.encoder)

        if "efficientvit" in self._model_name:
            import types

            self.encoder.forward_intermediates = types.MethodType(
                _efficientvit_forward_intermediates, self.encoder
            )

        data_config = resolve_data_config({}, model=self.encoder)
        logger.info("Encoder data config: %s", data_config)
        reset_timm_head(self.encoder)
        enc_channels = [cfg.encoder.in_chans] + get_timm_feature_channels(self.encoder)
        print(f"Multiscale Feature Channels: {enc_channels}")
        if "maxvit" in self._model_name:
            enc_strides = [1, 2, 4, 8, 16, 32]
        else:
            enc_strides = [1, 4, 8, 16, 32]

        # normalization config
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

        # Neck
        neck_kwargs = OmegaConf.to_object(cfg.neck)
        neck_name = neck_kwargs.pop("name")
        if neck_name is None:
            self.neck = None
        elif neck_name == "factorized_fpn":
            self.neck = FactorizedFPN2d(enc_channels[1:], **neck_kwargs)
        else:
            raise ValueError
        print("Before neck feature channels:", enc_channels)
        if self.neck is not None:
            enc_channels = [cfg.encoder.in_chans] + self.neck.output_channels
        print("After neck feature channels:", enc_channels)

        # Decoder
        dec_channels = [enc_channels[-1]] + cfg.decoder.channels[: cfg.decoder.n_blocks]
        self.dec_blocks = nn.ModuleList()
        for i in range(cfg.decoder.n_blocks):
            dec_block = UnetBlock(
                dec_channels[i], enc_channels[-i - 2], dec_channels[i + 1]
            )
            self.dec_blocks.append(dec_block)
        self.drop = nn.Dropout2d(cfg.decoder.dropout)
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], dec_channels[-1], 3, padding=1),
            LayerNorm2d(dec_channels[-1]),
            nn.GELU(),
            nn.Conv2d(dec_channels[-1], 1, 1),
        )

        ########################
        # direct regression head
        if cfg.reg_head.enable:
            self._enable_reg_head = True
            self.reg_feature_mode = cfg.reg_head.feature_mode
            self.reg_feature_idx = cfg.reg_head.feature_idx

            if self.reg_feature_mode == "enc":
                reg_channels = enc_channels
            elif self.reg_feature_mode == "dec":
                reg_channels = dec_channels
            elif self.reg_feature_mode == "enc_dec":
                reg_channels = []
                for i in range(0, cfg.decoder.n_blocks + 1):
                    if i == 0:
                        assert enc_channels[-i - 1] == dec_channels[0]
                        reg_channel = dec_channels[0]
                    else:
                        reg_channel = enc_channels[-i - 1] + dec_channels[i]
                    reg_channels.append(reg_channel)
            else:
                raise ValueError
            reg_channels = reg_channels[::-1]  # fine to coarse order
            reg_feature_stride = enc_strides[self.reg_feature_idx]
            logger.info(
                "\nEncoder channels:%s\n Decoder channels:%s\n Regress input channels:%s\n Regress feature stride:%d",
                enc_channels,
                dec_channels,
                reg_channels,
                reg_feature_stride,
            )

            final_norm_layer = partial(
                get_norm_layer(cfg.reg_head.norm),
                eps=cfg.reg_head.norm_eps,
            )
            if cfg.reg_head.type == "decoupled":
                self.reg_head = DecoupledRegressionHead(
                    in_features=reg_channels[self.reg_feature_idx],
                    cls_out_dim=cfg.num_kpts,
                    reg_out_dim=cfg.num_kpts * 2,
                    hidden_size=cfg.reg_head.hidden_dim,
                    pool_type=cfg.reg_head.pool_type,
                    first_drop_rate=cfg.reg_head.hidden_first_dropout,
                    reg_drop_rate=cfg.reg_head.reg_dropout,
                    cls_drop_rate=cfg.reg_head.cls_dropout,
                    norm_layer=final_norm_layer,
                    act_layer=cfg.reg_head.hidden_act,
                )
            elif cfg.reg_head.type == "decoupled_v2":
                self.reg_head = DecoupledRegressionHeadV2(
                    in_features=reg_channels[self.reg_feature_idx],
                    num_kpts=cfg.num_kpts,
                    feature_h=global_cfg.data.patch_size[1] // reg_feature_stride,
                    feature_w=global_cfg.data.patch_size[2] // reg_feature_stride,
                    hidden_size_per_kpt=cfg.reg_head.hidden_size_per_kpt,
                    hidden_size=cfg.reg_head.hidden_dim,
                    first_drop_rate=cfg.reg_head.hidden_first_dropout,
                    reg_drop_rate=cfg.reg_head.reg_dropout,
                    cls_drop_rate=cfg.reg_head.cls_dropout,
                    norm_layer=final_norm_layer,
                    act_layer=cfg.reg_head.hidden_act,
                )
            else:
                raise ValueError
        else:
            self._enable_reg_head = False

        if cfg.dsnt.enable:
            self._enable_dsnt = True
            # DSNT heatmap to coordinate
            heatmap_stride = enc_strides[-len(dec_channels)]
            heatmap_size = [e // heatmap_stride for e in global_cfg.data.patch_size[1:]]
            self.dsnt = DSNT2d(*heatmap_size, act=cfg.decoder.act)
        else:
            self._enable_dsnt = False

        # INIT WEIGHT
        # just an approximation, never mind
        pos_rates = torch.tensor(5e-5)
        bias_init = -torch.log(1.0 / pos_rates - 1.0)
        nn.init.constant_(self.seg_head[-1].bias, bias_init)
        logger.info("Init bias value of segmentation head to %s", bias_init)

    def forward(self, x):
        x = (x - self.mean) / self.std

        if "coat_" in self._model_name:
            enc_features = self.encoder.forward_features(x)
            enc_features = list(enc_features.values())
            # print([e.shape for e in enc_features])
            # print('has nan:', any([torch.isnan(e).any().item() for e in enc_features]))
        elif "convnext" in self._model_name:
            # convnext: stem has feature index 0
            # after stage 1, resolution is not reduce
            # so we use output of stage 1 as stride 4 features (first downscale feature)
            enc_features = self.encoder.forward_intermediates(
                x,
                indices=[1, 2, 3, 4],
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        elif (
            "maxvit" in self._model_name
            or "coatnet" in self._model_name
            or "swinv2" in self._model_name
            or "efficientvit" in self._model_name
        ):
            enc_features = self.encoder.forward_intermediates(
                x,
                indices=5 if "maxvit" in self._model_name else 4,
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        elif "efficientnet" in self._model_name:
            enc_features = self.encoder.forward_intermediates(
                x,
                indices=5,
                norm=False,
                stop_early=True,
                output_fmt="NCHW",
                intermediates_only=True,
                extra_blocks=False,
            )
        else:
            raise AssertionError

        if self.neck is not None:
            enc_features = self.neck(enc_features)

        dec_feature = enc_features[-1]
        dec_features = [dec_feature]
        for i, dec_block in enumerate(self.dec_blocks):
            dec_feature = dec_block(dec_feature, enc_features[-i - 2])
            dec_features.append(dec_feature)
        dec_features[-1] = dec_feature
        dec_features = dec_features[::-1]
        heatmap = self.seg_head(self.drop(dec_feature))

        if self._enable_dsnt:
            # DSNT
            dsnt_kpt = self.dsnt(heatmap[:, : self._num_kpts])
        else:
            dsnt_kpt = None

        # Direct Regression
        if self._enable_reg_head:
            if self.reg_feature_idx == -1:
                reg_feature = enc_features[-1]
            elif self.reg_feature_mode == "enc":
                reg_feature = enc_features[self.reg_feature_idx]
            elif self.reg_feature_mode == "dec":
                reg_feature = dec_features[self.reg_feature_idx]
            elif self.reg_feature_mode == "enc_dec":
                reg_feature = torch.cat(
                    [
                        enc_features[self.reg_feature_idx],
                        dec_features[self.reg_feature_idx],
                    ],
                    dim=1,
                )
            kptness, kpt = self.reg_head(reg_feature)
            kpt = kpt.reshape(-1, self._num_kpts, 2)
        else:
            kpt, kptness = None, None

        return kpt, kptness, heatmap, dsnt_kpt


if __name__ == "__main__":
    yaml_str = """

data:
    patch_size: [768, 768]

model:
    _target_: abc

    num_kpts: 1

    encoder:
        model_name: maxvit_tiny_tf_512.in1k
        pretrained: False
        img_size: ${data.patch_size}
        in_chans: 3
        # num_classes: 10
        # drop_rate: 0.1
        # drop_path_rate: 0.1

    # encoder:
    #     model_name: coat_lite_medium_384.in1k
    #     pretrained: False
    #     img_size: [384, 192]
    #     in_chans: 3
    #     num_classes: 10
    #     return_interm_layers: True
    #     out_features: [x1_nocls, x2_nocls, x3_nocls, x4_nocls]
    #     # drop_rate: 0.1
    #     # drop_path_rate: 0.1

    neck:
        name: factorized_fpn
        intermediate_channels_list: 64
        target_level: -2
        conv_ksizes: [3, 3]
        norm: layernorm_2d
        act: gelu
        interpolation_mode: bilinear

    decoder:
        n_blocks: 3
        channels: [256, 192, 128, 96, 64]
        dropout: 0.0
        out_channels: 1
        act: sigmoid

    reg_head:
        type: decoupled_v2
        feature_mode: enc_dec
        feature_idx: -1
        hidden_dim: 1024
        hidden_act: gelu
        hidden_first_dropout: 0.0
        cls_dropout: 0.0
        reg_dropout: 0.0
        norm: layernorm2d
        norm_eps: 1e-6
        hidden_size_per_kpt: 1

"""
    import logging

    logging.basicConfig(level=logging.INFO)
    global_cfg = OmegaConf.create(yaml_str)
    print(global_cfg)
    model = Unet2dModel(global_cfg).eval()
    print(model)

    with torch.inference_mode():
        inp = torch.rand((2, 3, 768, 768))
        out = model(inp)
        print("Input:", inp.shape)
        # print("Ouput:", {k: v.shape for k, v in out.items()})
        print([v.shape for v in out])
