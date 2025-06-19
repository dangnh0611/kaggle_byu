import math
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.ops import StochasticDepth
from yagm.layers.common import MaskedSoftmax1d

__all__ = [
    "get_global_pool",
    # pool 2d
    "GEMPooling2d",
    # global pool 2d
    "GlobalGEMPooling2d",
    # pool 1d
    "GEMPooling1d",
    # global pool 1d
    "GlobalGEMPooling1d",
    # global masked pool 1d
    "GlobalMaskedAvgPooling1d",
    "GlobalMaskedGEMPooling1d",
    "GlobalMaskedMaxPooling1d",
    "ChannelLastGlobalMaskedMaxPooling1d",
    "GlobalMaskedAttentionPooling1d",
    "GlobalMaskedConcatAttentionPooling1d",
]


class GEMPooling2d(nn.Module):
    """Generalized Mean Pooling 2D
    Ref: https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, ksize=(2, 2), p=3, eps=1e-6):
        super().__init__()
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C, H2, W2) where H2, W2 is based on ksize, usually H2 = H // ksize
        """
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), self.ksize).pow(
            1.0 / self.p
        )


class GlobalGEMPooling2d(nn.Module):
    """Generalized Mean Pooling 2D
    Ref: https://amaarora.github.io/posts/2020-08-30-gempool.html
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)
        Returns:
            (N, C)
        """
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class GEMPooling1d(nn.Module):
    """Generalized Mean Pooling 1D
    Ref: https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """

    def __init__(self, ksize=2, p=3, eps=1e-6):
        super().__init__()
        self.ksize = ksize
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C, L2) where L2 is based on ksize, usually L2 = L // ksize
        """
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), self.ksize).pow(
            1.0 / self.p
        )


class GlobalGEMPooling1d(nn.Module):
    """Generalized Mean Pooling 1D
    Ref: https://www.kaggle.com/code/scaomath/g2net-1d-cnn-gem-pool-pytorch-train-inference
    """

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C)
        """
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), x.size(-1)).pow(
            1.0 / self.p
        )


class GlobalMaskedAvgPooling1d(nn.Module):
    """Global Masked Average Pooling 1D
    Ref: https://github.com/amedprof/Feedback-Prize--English-Language-Learning/blob/main/src/model_zoo/pooling.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if mask is None:
            return F.adaptive_avg_pool1d(x, 1)
        padding_mask_expanded = mask.unsqueeze(1).expand(x.size())  # NCL
        sum_embeddings = torch.sum(x * padding_mask_expanded, -1)  # NCL -> NC
        # @TODO (dangnh): optimize
        sum_mask = padding_mask_expanded.sum(-1)  # NCT -> NC
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask  # NC
        return mean_embeddings


class GlobalMaskedGEMPooling1d(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self._avg_pool = GlobalMaskedAvgPooling1d()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        x = x.clamp(min=self.eps).pow(self.p)
        x = self._avg_pool(x, mask)
        x = x.pow(1.0 / self.p)
        return x


class GlobalMaskedMaxPooling1d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if mask is None:
            return F.adaptive_max_pool1d(x, 1)
        padding_mask_expanded = (
            torch.logical_not(mask).unsqueeze(1).expand(x.size())
        )  # NCL
        hidden_state_copy = x.clone()
        hidden_state_copy.masked_fill(padding_mask_expanded, float("-inf"))
        return torch.max(hidden_state_copy, -1)[0]  # NCL -> NC

class ChannelLastGlobalMaskedMaxPooling1d(GlobalMaskedMaxPooling1d):

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L, C)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        x = x.permute(0, 2, 1) # NLC -> NCL
        return super().forward(x, mask)
    

class GlobalMaskedAttentionPooling1d(nn.Module):
    """
    Global Masked Attention Pooling 1D
    Paper: https://arxiv.org/pdf/2008.01077v1.pdf
    Ref: https://github.com/daniel-code/TubeViT/blob/main/tubevit/model.py
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax1d(-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (N, C, L) if `channel_first` else (N, L, C) (default)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, C)
        """
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        _weight_logit = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_weight_logit, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


class GlobalMaskedConcatAttentionPooling1d(nn.Module):
    """
    Concat of (Attention Pooling + CLS) Pooling
    """

    def __init__(self, input_dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        self.proj = nn.Linear(input_dim, 1)
        self.masked_softmax = MaskedSoftmax1d(-1)

    def forward(self, input, mask=None):
        """
        Args:
            x: (N, C, L) if `channel_first` else (N, L, C) (default)
            mask: (N, L) where pad=False, nonpad=True
        Returns:
            (N, 2C)
        """
        if self.channel_first:
            # NCL -> NLC
            x = x.permute(0, 2, 1)
        x = input[:, 1:, ...]
        mask = mask[:, 1:]
        _weight_logit = self.proj(x).squeeze(dim=-1)
        att_w = self.masked_softmax(_weight_logit, mask).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        x = torch.cat([input[:, 0, :], x], axis=-1)  # N x (2C)
        return x


def get_global_pool(pool_type, channel_first=True, **kwargs):
    if pool_type == "avg_1d":
        if channel_first:
            return partial(nn.AdaptiveAvgPool1d, 1)
        raise NotImplementedError
    elif pool_type == "max_1d":
        if channel_first:
            return partial(nn.AdaptiveMaxPool1d, 1)
        raise NotImplementedError
    elif pool_type == "masked_avg_1d":
        if channel_first:
            return GlobalMaskedAvgPooling1d
        raise NotImplementedError
    elif pool_type == "masked_max_1d":
        if channel_first:
            return GlobalMaskedMaxPooling1d
        else:
            return ChannelLastGlobalMaskedMaxPooling1d
    elif pool_type == "masked_gem_1d":
        if channel_first:
            return partial(GlobalGEMPooling1d, p=kwargs["p"], eps=kwargs["eps"])
        raise NotImplementedError
    elif pool_type == "masked_attn_1d":
        return partial(
            GlobalMaskedAttentionPooling1d,
            input_dim=kwargs["input_dim"],
            channel_first=channel_first,
        )
    elif pool_type == "masked_concat_attn_1d":
        return partial(
            GlobalMaskedConcatAttentionPooling1d,
            input_dim=kwargs["input_dim"],
            channel_first=channel_first,
        )
    else:
        raise NotImplementedError
