import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=11, pretrained=False):
    resnet = models.resnet18(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet34(num_classes=11, pretrained=False):
    resnet = models.resnet34(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet50(num_classes=11, pretrained=False):
    resnet = models.resnet50(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_vgg11(num_classes=11):
    return models.vgg11(num_classes=num_classes)

def get_vgg13(num_classes=11):
    return models.vgg13(num_classes=num_classes)

def get_vgg16(num_classes=11):
    return models.vgg16_bn(num_classes=num_classes)

def get_vgg19(num_classes=11):
    return models.vgg19(num_classes=num_classes)

def get_densenet121(pretrained=False, **kwargs):
    return models.densenet121(pretrained, **kwargs)

def get_densenet161(pretrained=False, **kwargs):
    return models.densenet161(pretrained, **kwargs)

def get_densenet201(pretrained=False, **kwargs):
    return models.densenet201(pretrained, **kwargs)

# Add more here if you want...