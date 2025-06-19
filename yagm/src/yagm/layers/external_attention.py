"""
External Attention Modules.
You should give a star for this awesome work as well: https://github.com/xmu-xiaoma666/External-Attention-pytorch
"""

from torch import nn
from torch.nn import init

__all__ = ["SEAttention1d", "ECAAttention1d"]


class SEAttention1d(nn.Module):
    """Squeeze-Excitation Attention 1D
    Ref: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    """

    def __init__(self, inp_dim, hidden_dim, hidden_act=nn.ReLU, scale_act=nn.Sigmoid):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim, bias=False),
            hidden_act(inplace=True),
            nn.Linear(hidden_dim, inp_dim, bias=False),
            scale_act(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C, L)
        """
        y = self.gap(x).squeeze(-1)  # NC
        y = self.fc(y).unsqueeze(-1)  # NC -> NC -> NC1
        return x * y.expand_as(x)  # NCL


class ECAAttention1d(nn.Module):
    """Efficient Channel Attention 1D
    Ref: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/ECAAttention.py
    """

    def __init__(self, kernel_size=3, bias=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias
        )
        self.act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (N, C, L)
        Returns:
            (N, C, L)
        """
        # NCT
        y = self.gap(x).permute(0, 2, 1)  # NCL -> NC1 -> N1C
        y = self.act(self.conv(y)).permute(0, 2, 1)  # N1C -> N1C -> NC1
        return x * y.expand_as(x)  # NCL
