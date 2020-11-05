import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        # 卷积层
        self.conv_unit = nn.Sequential(
            # x[batch_size,3,32,32]=>[batch_size,6,]
            # 卷积
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 池化
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # flatten
        # 全连接层
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # x[b,3,32,32]==>[b,16,5,5]
        # 先卷积
        x = self.conv_unit(x)
        # 拉平
        x = x.view(batch_size, 16 * 5 * 5)
        # 全连接
        logits = self.fc_unit(x)

        return logits


