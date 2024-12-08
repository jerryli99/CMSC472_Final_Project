from .predefined import *
# from .custom import get_custom_model #uncomment this or modify this

Models = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "vgg11": get_vgg11,
    "vgg13": get_vgg13,
    "vgg16": get_vgg16,
    "vgg19": get_vgg19,
    "densenet121": get_densenet121,
    "densenet161": get_densenet161,
    "densenet201": get_densenet201,
    "vit16_224": get_vit16_224,
    # "custom": get_custom_model,  # Add your custom model here
}

def get_model(model_name, num_classes, pretrained=False):
    if model_name not in Models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(Models.keys())}")
    return Models[model_name](num_classes=num_classes, pretrained=pretrained)
