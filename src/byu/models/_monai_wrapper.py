import copy

import hydra
import torch
from lightning.pytorch.loggers import WandbLogger
from monai.networks.nets import flexible_unet, resnet
from omegaconf import OmegaConf
from torch import nn


class MonaiWrapper(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg_dict = OmegaConf.to_container(global_cfg.model, resolve=True)
        del cfg_dict["_target_"]
        model_cls = hydra.utils.get_class(cfg_dict.pop("_model_"))
        weight_init_method = cfg_dict.pop("weight_init")
        model: flexible_unet.FlexibleUNet = model_cls(**cfg_dict)
        print("INIT DECODER/HEAD..")
        self._init_weights(
            model.decoder,
            model.segmentation_head,
            global_cfg,
            method=weight_init_method,
        )
        self.model = model

    def forward(self, x):
        return [self.model(x)]

    def _init_weights(
        self, decoder: nn.Module, head: nn.Module, global_cfg, method="none"
    ):
        print("WEIGHT INIT:", method)
        if method == "none":
            pass
        elif method == "medicalnet":
            # https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py
            for m in decoder.modules():
                if isinstance(m, nn.Conv3d):
                    m.weight = nn.init.kaiming_normal(m.weight, mode="fan_out")
                elif isinstance(m, nn.BatchNorm3d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif method == "tf":
            from yagm.layers.weight_init import tf_weight_init

            decoder.apply(tf_weight_init)
        elif method == "last_zero":
            last_conv = head[0]
            assert "conv" in last_conv.__class__.__name__.lower()
            torch.nn.init.constant_(last_conv.weight, 0.0)  # Set weights to 0
            torch.nn.init.constant_(last_conv.bias, 0.0)  # Set biases to 0
            # Check the initialization
            print("Weights after initialization:")
            print(last_conv.weight)
            print("Biases after initialization:")
            print(last_conv.bias)
        elif method == "last_negative":
            last_conv = head[0]
            assert "conv" in last_conv.__class__.__name__.lower()
            torch.nn.init.constant_(last_conv.weight, 0.0)
            torch.nn.init.constant_(last_conv.bias, -1e3)
            # Check the initialization
            print("Weights after initialization:")
            print(last_conv.weight)
            print("Biases after initialization:")
            print(last_conv.bias)
        elif method == "optimal_bias":
            import segmentation_models_pytorch_3d as smp

            smp.base.initialization.initialize_decoder_bugfix(decoder)

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

            smp.base.initialization.initialize_head_bugfix(
                head,
                weight_method="xavier",
                bias=_cal_optimal_bias(),
            )
        else:
            raise ValueError


def _check_nan_or_inf(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        return torch.isnan(tensor_or_list).any() or torch.isinf(tensor_or_list).any()
    else:
        return any(
            [
                torch.isnan(e).any() or torch.isinf(e).any()
                for e in tensor_or_list
                if e is not None
            ]
        )


def check_nan_hook(module: nn.Module, input, output):
    """
    This function is used as a hook to check if any NaN values exist in the output of a layer.
    - `module`: the layer that the hook is registered on.
    - `input`: the input tensor to the layer.
    - `output`: the output tensor from the layer.
    """
    if _check_nan_or_inf(output) and not _check_nan_or_inf(input):
        data = {"input": input, "module": module.cpu(), "output": output}
        import pickle

        with open("nan_check.pkl", "wb") as f:
            pickle.dump(data, f)

        raise Exception(
            f"NaN detected in output of {module}, class={module.__class__}, is_training={module.training}, input_type={type(input)}, output_type={type(output)}!"
        )


# torch.nn.modules.module.register_module_forward_hook(check_nan_hook)
