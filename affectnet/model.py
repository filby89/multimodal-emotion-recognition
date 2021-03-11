import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import torchvision.models as models
import os


class AffectNet(BaseModel):
    def __init__(self):
        super(AffectNet, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        modules = list(resnet50.children())[:-1]      # delete the last fc layer.
        self.features = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 8))


    def forward(self, image):
        features = self.features(image).squeeze(-1).squeeze(-1)

        output = {}
        output['categorical'] = self.classifier(features)

        return output


