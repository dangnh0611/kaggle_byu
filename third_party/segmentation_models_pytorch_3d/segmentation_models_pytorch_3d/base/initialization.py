import torch.nn as nn


# Legacy code which not work for 3D Decoder
def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_decoder_bugfix(module):
    for m in module.modules():
        cls_name = m.__class__.__name__.lower()

        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif "batchnorm" in cls_name:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif "linear" in cls_name:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head_bugfix(module, weight_method="xavier", bias=0.0):
    assert weight_method in ["xavier", "zeros"]
    for m in module.modules():
        cls_name = m.__class__.__name__.lower()
        if "linear" in cls_name or "conv" in cls_name:
            if weight_method == "zeros":
                nn.init.constant_(m.weight, 0.0)
            elif weight_method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                if isinstance(bias, float):
                    nn.init.constant_(m.bias, bias)
                elif hasattr(bias, '__len__'):
                    assert len(bias) == len(m.bias)
                    for i in range(len(m.bias)):
                        nn.init.constant_(m.bias[i], bias[i])
                else:
                    raise ValueError
                print('HEAD BIAS AFTER INIT:\n', m.bias)
