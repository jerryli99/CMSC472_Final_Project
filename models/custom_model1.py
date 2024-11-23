# feel free to implement what network architechture you like
# this is just some backbone structure for the code..
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super(MyCustomModel, self).__init__()
        # ...
        # ...
    def forward(self, x):
        return x


def get_custom_model(num_classes, **kwargs):
    return MyCustomModel(num_classes=num_classes)