# src/bayesian_lora/models/resnet_cifar.py
import torch
import torch.nn as nn
from torchvision.models import resnet

class ResNetCIFAR(nn.Module):
    """
    Generic ResNet for CIFAR with configurable depth.
    """
    def __init__(self, depth: int = 18, num_classes: int = 10):
        super().__init__()
        
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = resnet.BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = resnet.BasicBlock
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        self.backbone = resnet.ResNet(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=nn.BatchNorm2d,
        )
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

