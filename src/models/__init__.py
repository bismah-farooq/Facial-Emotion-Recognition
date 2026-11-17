from .custom_cnn import CustomCNN
from .resnet_fer import ResNetFER


def build_model(name: str, num_classes: int = 7, pretrained: bool = True):
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet18":
        return ResNetFER(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'. Choose custom_cnn or resnet18.")
