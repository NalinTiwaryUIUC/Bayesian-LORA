# src/bayesian_lora/models/resnet_cifar.py
import torch
import torch.nn as nn
from torchvision.models import resnet

class ResNet18CIFAR(nn.Module):
    """
    CIFAR ResNet-18:
      - 3x3 conv1 (stride=1, padding=1)
      - no initial maxpool
      - BasicBlock layers [2,2,2,2]
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet.ResNet(
            block=resnet.BasicBlock,
            layers=[2, 2, 2, 2],
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

class ResNet34CIFAR(nn.Module):
    """ResNet-34 adapted for CIFAR (32×32)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = resnet.ResNet(
            block=resnet.BasicBlock,
            layers=[3, 4, 6, 3],
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