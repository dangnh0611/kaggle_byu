import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def resolve_optimizer_kwargs(cfg):
    kwargs = dict(cfg)
    return kwargs


def create_optimizer(model: nn.Module, cfg):
    optim_kwargs = resolve_optimizer_kwargs(cfg)
    name = optim_kwargs.pop("name")
    logger.info("Optim kwargs: %s", optim_kwargs)

    # @TODO - support weight decay for norm,... properly
    # for example: https://github.com/IDEA-Research/MaskDINO/blob/main/train_net.py
    # norm_module_types = (
    #         torch.nn.BatchNorm1d,
    #         torch.nn.BatchNorm2d,
    #         torch.nn.BatchNorm3d,
    #         torch.nn.SyncBatchNorm,
    #         # NaiveSyncBatchNorm inherits from BatchNorm2d
    #         torch.nn.GroupNorm,
    #         torch.nn.InstanceNorm1d,
    #         torch.nn.InstanceNorm2d,
    #         torch.nn.InstanceNorm3d,
    #         torch.nn.LayerNorm,
    #         torch.nn.LocalResponseNorm,
    # )
    logger.warning("PLEASE SUPPORT WEIGHT DECAY/PARAMETER GROUPS PROPERLY!")

    if name.startswith("timm@"):
        optim_kwargs["opt"] = name.replace("timm@", "")
        from timm.optim.optim_factory import create_optimizer_v2

        optimizer = create_optimizer_v2(model, **optim_kwargs)
    elif name.startswith("torch@"):
        name = name.replace("torch@", "")
        TORCH_OPTIMS = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        optim_cls = TORCH_OPTIMS[name]
        optimizer = optim_cls(model.parameters(), **optim_kwargs)
    elif name.startswith("apex@"):
        name = name.replace("apex@", "")
        import apex

        APEX_OPTIMS = {
            "lamb": apex.optimizers.FusedLAMB,
            "mixed_lamb": apex.optimizers.FusedMixedPrecisionLamb,
        }
        optim_cls = APEX_OPTIMS[name]
        optimizer = optim_cls(model.parameters(), **optim_kwargs)
    else:
        raise ValueError

    return optimizer
