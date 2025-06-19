from torch import nn
import logging

logger = logging.getLogger(__name__)


def get_timm_feature_channels(model: nn.Module):
    """Return a list of feature channels of a timm model"""
    if hasattr(model, "feature_info"):
        logger.info("Feature Info:\n%s", model.feature_info)
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


def reset_timm_head(model: nn.Module):
    """Reset classifier attached to a timm model"""
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=0)
    if hasattr(model, "prune_intermediate_layers"):
        model.prune_intermediate_layers(
            prune_norm=True,
            prune_head=True,
        )
