import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

from masker.models.simple_cnn import SimpleMaskRCNN
from masker.models.unet import UNet

def create_model():
    # net = SimpleTest()
    net = UNet(1, 1, 8)
    # net = SimpleViT(
    #     image_size=1024,
    #     patch_size=32,
    #     num_classes=1,
    #     dim=1024*2,
    #     depth=3,
    #     heads=16*4,
    #     mlp_dim=2048,
    #     channels=1
    # )

    #net = SimpleMaskRCNN()

    return net