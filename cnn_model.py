import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights

class Model(nn.Module):
    def __init__(self,num_classes = 7):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.model = self.__load_model__()

    def __load_model__(self):
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.features[0] = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280,self.num_classes)
        )
        return model

    def forward(self,input_data):
        return self.model(input_data)
