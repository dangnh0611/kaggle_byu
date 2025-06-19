import logging

import torch
from monai.networks.nets import swin_unetr
from monai.networks.utils import copy_model_state
from omegaconf import OmegaConf
from torch import nn

logger = logging.getLogger(__name__)


class SwinUNETR(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg_dict = OmegaConf.to_container(global_cfg.model, resolve=True)
        del cfg_dict["_target_"]
        pretrained_path = cfg_dict.pop("pretrained")
        model = swin_unetr.SwinUNETR(**cfg_dict)
        # resource = (
        #     "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
        # )
        # resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
        # ssl_weights_path = "./ssl_pretrained_weights.pth"
        # download_url(resource, ssl_weights_path)

        if pretrained_path is not None:
            logger.info("Loading pretrain weight from %s", pretrained_path)
            state_dict = torch.load(pretrained_path)
            print(state_dict.keys())
            if "state_dict" in state_dict:
                # https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
                model.load_from(state_dict)
            else:
                # https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth
                state_dict = state_dict["model"]
                dst_dict, loaded, not_loaded = copy_model_state(
                    model, state_dict, filter_func=swin_unetr.filter_swinunetr
                )
                logger.info("Loaded: %s\nNot loaded: %s", loaded, not_loaded)

        # INIT HEAD BIAS
        def _cal_optimal_bias():
            # just an approximation, never mind
            POS_RATES = [
                4.6182e-05,
                None,
                2.3524e-05,
                3.7194e-04,
                1.6056e-04,
                3.0082e-05,
            ]
            low, high = global_cfg.data.label_smooth
            pos_rates = torch.tensor(
                [POS_RATES[idx] for idx in global_cfg.data.particles]
            )
            pos_rates = low + torch.tensor(pos_rates) / 0.95 * (high - low)
            init_bias = -torch.log(1.0 / pos_rates - 1.0)
            return init_bias.tolist()

        import segmentation_models_pytorch_3d as smp

        smp.base.initialization.initialize_head_bugfix(
            model.out.conv.conv,
            weight_method="xavier",
            bias=_cal_optimal_bias(),
        )

        self.model = model

    def forward(self, x):
        return [self.model(x)]
