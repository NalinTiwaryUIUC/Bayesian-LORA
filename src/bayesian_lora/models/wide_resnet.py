# src/bayesian_lora/models/wide_resnet.py
import torch
import torch.nn as nn
from collections import OrderedDict

class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_norm", nn.BatchNorm2d(channels)),
            ("1_act", nn.ReLU(inplace=False)),
            ("2_conv", nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
            ("3_norm", nn.BatchNorm2d(channels)),
            ("4_act", nn.ReLU(inplace=False)),
            ("5_drop", nn.Dropout2d(dropout, inplace=False)),
            ("6_conv", nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return x + self.block(x)

class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_norm", nn.BatchNorm2d(in_channels)),
            ("1_act", nn.ReLU(inplace=False)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_conv", nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)),
            ("1_norm", nn.BatchNorm2d(out_channels)),
            ("2_act", nn.ReLU(inplace=False)),
            ("3_drop", nn.Dropout2d(dropout, inplace=False)),
            ("4_conv", nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.norm_act(x)
        return self.block(out) + self.downsample(x)

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth - 1)),
        )

    def forward(self, x):
        return self.block(x)

class WideResNet(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
        super().__init__()
        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_conv", nn.Conv2d(in_channels, self.filters[0], 3, stride=1, padding=1, bias=False)),
            ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("4_norm", nn.BatchNorm2d(self.filters[3])),
            ("5_act", nn.ReLU(inplace=False)),
            ("6_pool", nn.AvgPool2d(kernel_size=8)),
            ("7_flat", nn.Flatten()),
            ("8_fc", nn.Linear(in_features=self.filters[3], out_features=labels)),
        ]))
        self._initialize()

    def _initialize(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)

def WRN_28_10_CIFAR(num_classes=10):
    return WideResNet(depth=28, width_factor=10, dropout=0.3, in_channels=3, labels=num_classes)