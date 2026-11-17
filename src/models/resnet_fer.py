"""Model B - Transfer learning with ResNet-18 (ImageNet-pretrained).

Owner: <PARTNER NAME>

Adaptations for 48x48 grayscale FER images:
  * First conv: pretrained RGB 7x7 stride-2 kernel is replaced by a 3x3
    stride-1 conv whose weights are initialised from the centre of the
    pretrained kernel, summed over RGB -> 1 channel. This keeps more
    spatial resolution for tiny faces.
  * The initial max-pool is removed (48 -> 48 -> 24 -> 12 -> 6 after the
    four stages instead of collapsing to 2x2).
  * Final FC replaced with Dropout + Linear(512, 7).
Fine-tuning schedule (see train.py): the backbone is frozen for the first
`freeze_epochs` epochs (only the new head/first conv learn), then the whole
network is unfrozen and trained with a lower backbone learning rate.
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetFER(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        old = net.conv1.weight.data  # (64, 3, 7, 7)
        conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                conv1.weight.copy_(old[:, :, 2:5, 2:5].sum(dim=1, keepdim=True))
        net.conv1 = conv1
        net.maxpool = nn.Identity()
        net.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(net.fc.in_features, num_classes))
        self.net = net

    # Parameter groups so train.py can freeze / use differential learning rates.
    def head_parameters(self):
        return list(self.net.conv1.parameters()) + list(self.net.fc.parameters())

    def backbone_parameters(self):
        head = {id(p) for p in self.head_parameters()}
        return [p for p in self.net.parameters() if id(p) not in head]

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone_parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.net(x)
