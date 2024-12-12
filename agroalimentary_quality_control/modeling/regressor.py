from torch import nn
from torchvision import models


class RocketRegressor(nn.Module):
    def __init__(self, target_cols, final_dropout_p=0.2):
        super().__init__()

        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        last_channel_width = self.mobilenet.last_channel
        output_width = len(target_cols)

        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=final_dropout_p),
            nn.Linear(last_channel_width, output_width)
        )


    def forward(self, x):
        x = self.mobilenet(x)

        return x
