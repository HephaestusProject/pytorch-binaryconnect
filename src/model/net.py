import torch
import torch.nn as nn

from omegaconf import DictConfig
from torchsummary import summary as torch_summary

from src.nn.binarized_linear import BinaryLinear
from src.nn.binarized_conv2d import BinarizedConv2d

class BinaryLinear(nn.Module):
    def __init__(self, model_config: DictConfig):
        super(BinaryLinear, self).__init__()        
        mode = str(model_config.params.mode)
        classes = int(model_config.params.classes)
        dropout_ratio = float(model_config.params.dropout_ratio)        
        self.width = int(model_config.params.width)
        self.height = int(model_config.params.height)
        self.channels = int(model_config.params.channels)

        self.fc1 = BinaryLinear(width * height * channels, 1024, bias=False, mode="Stocastic")
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = BinaryLinear(1024, 1024, bias=False, mode=mode)
        self.batch2 = nn.BatchNorm1d(1024)
        self.fc3 = BinaryLinear(1024, 1024, bias=False, mode=mode)
        self.batch3 = nn.BatchNorm1d(1024)
        self.fc4 = BinaryLinear(1024, classes, bias=False, mode=mode)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.view(-1, self.width * self.height * self.channels)
        x = self.relu(self.fc1(x))
        x = self.batch1(x)
        x = self.relu(self.fc2(x))
        x = self.batch2(x)
        x = self.relu(self.fc3(x))
        x = self.batch3(x)
        x = self.fc4(x)        
        return x

    def summary(self):
        summary(self, input_size=(self.channels, self.height, self.width))


class BinaryConv(nn.Module):
    def __init__(self, model_config: DictConfig):
        super(BinaryConv, self).__init__()
        
        mode = str(model_config.params.mode)
        classes = int(model_config.params.classes)        
        self.width = int(model_config.params.width)
        self.height = int(model_config.params.height)
        self.channels = int(model_config.params.channels)

        self.conv1 = nn.BinarizedConv2d(in_channels=self.channels,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               mode=mode)
        self.batch1 = nn.BatchNorm2d(num_features=128)

        self.conv2 = nn.BinarizedConv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               mode=mode)
        self.batch2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.BinarizedConv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               mode=mode)
        self.batch3 = nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.BinarizedConv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True,
                               padding_mode='zeros',
                               mode=mode)
        self.batch4 = nn.BatchNorm2d(num_features=512)

        self.fc1 = nn.BinarizedLinear(16 * 5 * 5, 1024)
        self.fc2 = nn.BinarizedLinear(1024, 1024)
        self.fc3 = nn.BinarizedLinear(1024, classes)

        self.pool = nn.MaxPool2d(2, 2)        
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        x = self.relu(self.batch3(self.conv3(x)))
        x = self.relu(self.batch4(self.conv4(x)))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))  
        return x

    def summary(self):
        summary(self, input_size=(self.channels, self.height, self.width))

if __name__ == "__main__":
    from omegaconf import OmegaConf
    OmegaConf.create()
