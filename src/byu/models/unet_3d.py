import logging

import hydra
import segmentation_models_pytorch_3d as smp
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from segmentation_models_pytorch_3d.base import SegmentationHead
from segmentation_models_pytorch_3d.decoders.unet.decoder import MultiscaleUnetDecoder

logger = logging.getLogger(__name__)


# https://github.com/ZFTurbo/segmentation_models_pytorch_3d/blob/main/segmentation_models_pytorch_3d/decoders/unet/model.py
class Unet3dModel(nn.Module):

    def __init__(self, global_cfg):
        super().__init__()
        cfg = global_cfg.model

        # ========== ENCODER =============
        encoder_cls = hydra.utils.get_object(cfg.encoder._target_)
        self.encoder = encoder_cls(global_cfg)

        # ========== NECK ==========
        if getattr(cfg, "neck", None) is not None:
            neck_kwargs = OmegaConf.to_object(cfg.neck)
            neck_cls = hydra.utils.get_object(neck_kwargs.pop("_target_"))
            print("ENCODER OUTPUT CHANNELS:", self.encoder.output_channels)
            self.neck = neck_cls(self.encoder.output_channels[1:], **neck_kwargs)
        else:
            self.neck = None

        # ========== DECODER =============
        self.decoder = MultiscaleUnetDecoder(
            encoder_channels=(
                self.encoder.output_channels
                if self.neck is None
                else self.encoder.output_channels[0:1] + self.neck.output_channels
            ),
            decoder_channels=cfg.decoder.decoder_channels[: cfg.decoder.n_blocks],
            n_blocks=cfg.decoder.n_blocks,
            use_batchnorm=cfg.decoder.use_batchnorm,
            attention_type=cfg.decoder.attention_type,
            center=cfg.decoder.center,
            strides=OmegaConf.to_object(cfg.decoder.strides)[: cfg.decoder.n_blocks][
                ::-1
            ],
        )

        # =========== MULTISCALE SEGMENTATION HEAD =========
        ms_idxs = OmegaConf.to_object(cfg.head.ms)
        if not isinstance(ms_idxs, list):
            assert isinstance(ms_idxs, int) and ms_idxs >= 0
            ms_idxs = [ms_idxs]
        self.ms_idxs = ms_idxs
        self.ms_seg_heads = nn.ModuleList()
        for ms_idx in ms_idxs:
            self.ms_seg_heads.append(
                SegmentationHead(
                    in_channels=self.decoder.output_channels[ms_idx],
                    out_channels=cfg.head.out_channels,
                    activation=None,
                    kernel_size=3,
                )
            )

        # ===== WEIGHT INIT =====
        # legacy/early code using https://github.com/ZFTurbo/segmentation_models_pytorch_3d/blob/main/segmentation_models_pytorch_3d/base/initialization.py
        # which as not patched to work for 3D Decoder, instead for 2D Decoder only
        # so this is a bug -> no initialization was performed
        if cfg.decoder.weight_init is None:
            smp.base.initialization.initialize_decoder(self.decoder)
        elif cfg.decoder.weight_init == "xavier":
            smp.base.initialization.initialize_decoder_bugfix(self.decoder)
        else:
            raise ValueError

        for seg_head in self.ms_seg_heads:
            if cfg.head.weight_init is None:
                # use default bias=0
                smp.base.initialization.initialize_head(seg_head)
            else:

                def _cal_optimal_bias():
                    low, high = global_cfg.data.label_smooth
                    # just an approximation, never mind
                    pos_rates = torch.tensor([5e-5])
                    pos_rates = low + torch.tensor(pos_rates) / 0.95 * (high - low)
                    init_bias = -torch.log(1.0 / pos_rates - 1.0)
                    return init_bias.tolist()

                smp.base.initialization.initialize_head_bugfix(
                    seg_head,
                    weight_method=cfg.head.weight_init,
                    bias=_cal_optimal_bias(),
                )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        if self.neck is not None:
            features = [features[0]] + self.neck(features[1:])
        # print(
        #     "encoder:",
        #     [e.shape if isinstance(e, torch.Tensor) else e for e in features],
        # )

        decoder_outputs = self.decoder(*features)
        # print("decoder:", [e.shape for e in decoder_outputs])

        ms_masks = []
        for ms_idx, seg_head in zip(self.ms_idxs, self.ms_seg_heads):
            mask = seg_head(decoder_outputs[ms_idx])
            mask = mask.permute(0, 1, 4, 2, 3)  # BCHWD -> BCDHW
            ms_masks.append(mask)
        return ms_masks


if __name__ == "__main__":
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)

    yaml_str = """

data:
    label_smooth: [0.0, 1.0]
    patch_size: [64 448, 448]

model:

    _target_: byu.models.unet_3d.Unet3dModel
    head:
        ms: [0] 
        out_channels: 1
        weight_init: xavier

    # encoder:
    #     model_name: x3d
    #     model_size: M
    #     pretrained: ckpts/x3d_m.pyth

    # decoder:
    #     decoder_channels: [192, 96, 48, 24, 16]
    #     n_blocks: 5
    #     use_batchnorm: True
    #     attention_type: null
    #     center: True
    #     strides: [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]

    

    # encoder:
    #     model_name: mvitv2
    #     model_size: B
    #     pretrained: ckpts/MViTv2_B_32x3_k400_f304025456.pyth

    # decoder:
    #     decoder_channels: [768, 384, 192, 96]
    #     n_blocks: 4
    #     use_batchnorm: True
    #     attention_type: null
    #     center: True
    #     strides: [[4, 4, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]]

    

    encoder:
        _target_: byu.models.encoders.smp.SMPEncoder
        model_name: smp_resnet101

    decoder:
        decoder_channels: [192, 96, 64, 48, 32]
        n_blocks: 2
        use_batchnorm: True
        attention_type: scse
        center: False
        strides: [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        weight_init: xavier



"""
    device = torch.device("cpu")
    global_cfg = OmegaConf.create(yaml_str)

    print("GLOBAL CONFIG:")
    print(global_cfg)
    model = Unet3dModel(global_cfg)
    model.to(device).eval()
    print(model)

    # print(
    #     model.encoder.output_channels,
    #     model.encoder.feature_shapes,
    #     model.encoder.downsample_block_idxs,
    #     sep="\n",
    # )

    # exit()

    inputs = [
        torch.rand((2, 1, 64, 320, 320)).float(),
        # torch.rand((1, 1, 32, 320, 320)).float(),
        # torch.rand((1, 1, 32, 640, 640)).float(),
        # torch.rand((1, 1, 96, 96, 96)).float(),
        # torch.rand((1, 1, 128, 128, 128)).float(),
        # torch.rand((1, 1, 224, 224, 224)).float(),
    ]

    with torch.inference_mode():
        for i, inp in enumerate(inputs):
            print("INPUT:", inp.shape)
            outs = model(inp.to(device))
            print(len(outs), "OUTPUTS:")
            for out in outs:
                print("\t", out.shape, out.dtype)
                assert not (out.isnan().any() or out.isinf().any())
            print("------------------------------")
