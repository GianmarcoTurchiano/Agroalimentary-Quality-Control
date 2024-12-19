import torch
from torch import nn
from torchvision import models


class RocketRegressor(nn.Module):
    def __init__(
        self,
        pretrained_model_output_size,
        pretrained_model_path,
        target_cols,
        target_device,
        final_dropout_p=0.2
    ):
        super().__init__()

        self.model = models.mobilenet_v2(num_classes=pretrained_model_output_size)
        d = torch.load(pretrained_model_path, weights_only=True, map_location=target_device)
        self.model.load_state_dict(d['model'])

        last_channel_width = self.model.last_channel
        output_width = len(target_cols)

        self.model.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(last_channel_width, int(last_channel_width / 2)),
                nn.ReLU6(),
                nn.Linear(int(last_channel_width / 2), int(last_channel_width / 2)),
                nn.ReLU6(),
                nn.Linear(int(last_channel_width / 2), int(last_channel_width / 3)),
                nn.ReLU6(),
                nn.Dropout(p=final_dropout_p),
                nn.Linear(int(last_channel_width / 3), 1)
            ) 
            for _ in range(output_width)
        ])


    def forward(self, x):
        x = self.model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = [classifier(x) for classifier in self.model.classifier]
        x = torch.cat(x, dim=1)

        return x
