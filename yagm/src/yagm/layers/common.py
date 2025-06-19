from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "TransposeLast",
    "MaskedSoftmax1d",
    "MLP",
    "SameConv1d",
    "MaskedConv1d",
    "CausalConv1d",
    "Conv1dNormAct",
    "Conv1dActNorm",
    "make_divisible",
    "make_net_dims",
]


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class MaskedSoftmax1d(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L)
            mask: (N, L) where where pad=False, nonpad=True
        Returns:
            (N, L)
        """
        if mask is not None:
            x = x.masked_fill(~mask, torch.finfo(x.dtype).min)
        return F.softmax(x, dim=self.dim)


class MLP(torch.nn.Sequential):
    """Modify from torchvision's MLP, add option for last Norm/Activation/Dropout.
    This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels: Number of channels of the input
        hidden_channels: List of the hidden channel dimensions
        norm_layer: Norm layer that will be stacked on top of the linear layer.
            If ``None`` this layer won't be used. Default: ``None``
        activation_layer: Activation function which will be stacked on top of the normalization layer
            (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used.
            Default: `torch.nn.ReLU`
        inplace: Parameter for the activation layer, which can optionally do the operation in-place.
            Default is `None`, which uses the respective default values of the `activation_layer` and Dropout layer.
        bias: Whether to use bias in the linear layer. Default ``True``
        dropout: The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        act_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        last_norm=True,
        last_activation=True,
        last_dropout=True,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            is_not_last = i != len(hidden_channels) - 1
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None and (is_not_last or last_norm):
                layers.append(norm_layer(hidden_dim))
            if is_not_last or last_activation:
                layers.append(act_layer())
            if is_not_last or last_dropout:
                layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        super().__init__(*layers)


class SameConv1d(nn.Conv1d):
    """Convolution 1D layer without changing sequence length L.
    Input: (N, C1, L)
    Output: (N, C2, L), EXACTLY same sequence length as Input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        if dilation != 1:
            raise ValueError("dilation != 1 is currently not supported!")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _calc_same_pad(self, seq_len, ksize, stride):
        if seq_len % stride == 0:
            pad = max(ksize - stride, 0)
        else:
            pad = max(ksize - (seq_len % stride), 0)
        return pad

    def forward(self, x):
        pad = self._calc_same_pad(x.size(-1), self.kernel_size, self.stride)
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return super().forward(x)


class SameDepthwiseConv1d(SameConv1d):
    """Depthwise Convolution 1D layer without changing sequence length L.
    Input: (N, C1, L)
    Output: (N, C2, L), EXACTLY same sequence length as Input
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=True,
    ):
        assert out_channels % in_channels == 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


class MaskedConv1d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=17,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = SameConv1d(
            in_channels,
            out_channels,
            kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.supports_masking = True

    def _compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, :: self.stride]
        return mask

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(
                ~mask[:, None, :], torch.tensor(0.0, dtype=x.dtype, device=x.device)
            )
            mask = self._compute_mask(x, mask)
        x = self.conv(x)
        return x, mask


class CausalConv1d(nn.Sequential):
    """Causal Convolution 1D, where token at index i is only mixed with previous token 0..i-1
    This ensures CAUSALITY and sometime reduce the effect of padding (train-test gap) in CNN 1D model.
    Ref: https://www.kaggle.com/competitions/asl-signs/discussion/406684
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0.0),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding="valid",
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )


class Conv1dNormAct(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        act_layer=partial(nn.ReLU, inplace=True),
        norm_layer=partial(nn.BatchNorm1d),
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_layer is None),
        )
        if norm_layer is None:
            norm = nn.Identity()
        else:
            norm = norm_layer(out_channels)
        act = act_layer()
        super().__init__(conv, norm, act)


class Conv1dActNorm(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        act_layer=partial(nn.ReLU, inplace=True),
        norm_layer=partial(nn.BatchNorm1d),
    ):
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=(norm_layer is None),
        )
        act = act_layer()
        if norm_layer is None:
            norm = nn.Identity()
        else:
            norm = norm_layer(out_channels)
        super().__init__(conv, act, norm)


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_net_dims(depth, base_dim, dim_scale_method, width_multiplier=1, divisor=8):
    ret_dims = [base_dim]
    if "add" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "add"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(ret_dims[-1] + coef, divisor))
    elif "mul" in dim_scale_method:
        _method, coef = dim_scale_method.split("_")
        assert _method == "mul"
        coef = float(coef)
        for _ in range(depth):
            ret_dims.append(make_divisible(int(ret_dims[-1] * coef), divisor))
    else:
        raise ValueError
    ret_dims = [make_divisible(e * width_multiplier, divisor) for e in ret_dims]
    return ret_dims[1:]


class LSTMBlock2d(nn.Module):

    def __init__(self, in_chs, bidirectional=False, num_layers=1):
        """2D LSTM to encode a sequence of 2D images along time/depth dimension.
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


class TimeToChannelSlidingResample(nn.Module):

    def __init__(self, stride=1, channels=3, padding_mode="replicate"):
        """
        Sliding Resample along temporal/depth dimension,
        effectively convert time/depth to channels dimension.
        Useful for 2.5D image/voxel/video modeling
            Input shape: (B, T1, ...)
            Ouput shape: (B, T2, C, ...)
        """
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
