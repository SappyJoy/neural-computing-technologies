from torchvision.models import resnet50

from object_detector import ObjectDetector


def create_model(num_classes):
    resnet = resnet50(pretrained=True)
    # freeze all ResNet50 layers so they will *not* be updated during the
    # training process
    for param in resnet.parameters():
        param.requires_grad = False

    return ObjectDetector(resnet, num_classes)
