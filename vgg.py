'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from torch.nn import functional as F
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(
            self,
            features,
            num_classes= 1000,
            init_weights= True
    ):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, feature=False):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        if feature:
            return x
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg( _cfg, bn, **kwargs):
    model = VGG(make_layers(cfg[_cfg], batch_norm=bn), **kwargs)
    return model

def vgg11(num_classes):
    return _vgg('VGG11', False, num_classes=num_classes)
def vgg13(num_classes):
    return _vgg('VGG13', False, num_classes=num_classes)
def vgg16(num_classes):
    return _vgg('VGG16', False, num_classes=num_classes)
def vgg19(num_classes):
    return _vgg('VGG19', False, num_classes=num_classes)
def vgg11_bn(num_classes):
    return _vgg('VGG11', True, num_classes=num_classes)
def vgg13_bn(num_classes):
    return _vgg('VGG13', True, num_classes=num_classes)
def vgg16_bn(num_classes):
    return _vgg('VGG16', True, num_classes=num_classes)
def vgg19_bn(num_classes):
    return _vgg('VGG19', True, num_classes=num_classes)
def _test():
    net = vgg19(10)
    x = torch.randn(2,3,32,32)
    y = net(x, True)
    print(y.size())

#_test()