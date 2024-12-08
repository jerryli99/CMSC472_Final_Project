import torch.nn as nn
import timm
from torchvision import models

def get_resnet18(num_classes=20, pretrained=False):
    resnet = models.resnet18(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet34(num_classes=20, pretrained=False):
    resnet = models.resnet34(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet50(num_classes=20, pretrained=False):
    resnet = models.resnet50(pretrained=pretrained)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_vgg11(num_classes=20, pretrained=False):
    vgg = models.vgg11(pretrained=pretrained)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
    return vgg

def get_vgg13(num_classes=20, pretrained=False):
    vgg = models.vgg13(pretrained=pretrained)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
    return vgg

def get_vgg16(num_classes=20, pretrained=False):
    vgg = models.vgg16_bn(pretrained=pretrained)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
    return vgg

def get_vgg19(num_classes=20, pretrained=False):
    vgg = models.vgg19(pretrained=pretrained)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
    return vgg

def get_densenet121(num_classes=20, pretrained=True):
    densenet = models.densenet121(pretrained=pretrained)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    return densenet

def get_densenet161(num_classes=20, pretrained=True):
    densenet = models.densenet161(pretrained=pretrained)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    return densenet

def get_densenet201(num_classes=20, pretrained=True):
    densenet = models.densenet201(pretrained=pretrained)
    densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
    return densenet

def get_vit16_224(num_classes=20, pretrained=True):
    return timm.create_model('vit_base_patch16_224',
                             pretrained=True,
                             num_classes=num_classes)

# Add more here if you want...