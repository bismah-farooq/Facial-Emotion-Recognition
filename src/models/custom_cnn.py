"""Model A - Custom CNN trained from scratch.

Owner: Bismah Farooq

VGG-style network sized for 48x48 grayscale input:
four conv blocks (Conv-BN-ReLU x2 -> MaxPool -> Dropout) with
64/128/256/512 channels, then global average pooling and a small
fully-connected head. Spatial size: 48 -> 24 -> 12 -> 6 -> 3.
~4.8M parameters.
"""
import torch.nn as nn


def _block(c_in: int, c_out: int, drop: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, 64, 0.1),
            _block(64, 128, 0.2),
            _block(128, 256, 0.3),
            _block(256, 512, 0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
