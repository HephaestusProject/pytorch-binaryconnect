import torch
import torch.nn as nn
from omegaconf import DictConfig
from src.nn.binarized_conv2d import BinarizedConv2d
from src.nn.binarized_linear import BinarizedLinear
from torchsummary import summary as torch_summary


class BinaryLinear(nn.Module):
    """rThe MLP we train on MNIST consists in 3 hidden layers of 1024 Rectifier
    Linear Units (ReLU) [34, 24, 3] and a L2-SVM output layer (L2-SVM has been shown to perform
    better than Softmax on several classification benchmarks [30, 32]). The square hinge loss is minimized with SGD without momentum. We use an exponentially decaying learning rate. We use Batch
    Normalization with a minibatch of size 200 to speed up the training. 

    Refs: https://arxiv.org/pdf/1511.00363.pdf
    """

    def __init__(self, model_config: DictConfig):
        super(BinaryLinear, self).__init__()
        mode = str(model_config.params.mode)
        classes = int(model_config.params.classes)

        self.width = int(model_config.params.width)
        self.height = int(model_config.params.height)
        self.channels = int(model_config.params.channels)

        self.fc1 = BinarizedLinear(
            self.width * self.height * self.channels, 1024, bias=False, mode=mode
        )
        self.batch1 = nn.BatchNorm1d(1024)
        self.fc2 = BinarizedLinear(1024, 1024, bias=False, mode=mode)
        self.batch2 = nn.BatchNorm1d(1024)
        self.fc3 = BinarizedLinear(1024, classes, bias=False, mode=mode)

        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.width * self.height * self.channels)
        x = self.relu(self.batch1(self.fc1(x)))
        x = self.batch2(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.softmax(x)
        return x

    def summary(self):
        # torchsummary/torchsummary.py:48: AssertionError
        # device = device.lower()
        # assert device in [
        #    "cuda",
        #    "cpu",
        # ], "Input device is not valid, please specify 'cuda' or 'cpu'"
        # AssertionError: Input device is not valid, please specify 'cuda' or 'cpu'
        device = str(self.device).split(":")[0]
        torch_summary(
            self, input_size=(self.channels, self.height, self.width), device=device,
        )

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {
            buf.device for buf in self.buffers()
        }
        if len(devices) != 1:
            raise RuntimeError(
                "Cannot determine device: {} different devices found".format(len(devices))
            )
        return next(iter(devices))


class BinaryConv(nn.Module):
    """rWe preprocess the data using global contrast normalization
    and ZCA whitening. We do not use any data-augmentation (which can really be a game changer for
    this dataset [35]). 
    
    The architecture of our CNN is:
    (2×128C3)−MP2−(2×256C3)−MP2−(2×512C3)−MP2−(2×1024F C)−10SV M (5)

    Where C3 is a 3 × 3 ReLU convolution layer, MP2 is a 2 × 2 max-pooling layer, F C a fully
    connected layer, and SVM a L2-SVM output layer. 
    The square hinge loss is minimized with ADAM. We use an exponentially decaying learning
    rate. We use Batch Normalization with a minibatch of size 50 to speed up the training

    Refs: https://arxiv.org/pdf/1511.00363.pdf
    """

    def __init__(self, model_config: DictConfig):
        super(BinaryConv, self).__init__()

        mode = str(model_config.params.mode)
        classes = int(model_config.params.classes)
        self.width = int(model_config.params.width)
        self.height = int(model_config.params.height)
        self.channels = int(model_config.params.channels)

        self.conv1 = BinarizedConv2d(
            in_channels=self.channels,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch1 = nn.BatchNorm2d(num_features=128)

        self.conv2 = BinarizedConv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = BinarizedConv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch3 = nn.BatchNorm2d(num_features=256)

        self.conv4 = BinarizedConv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch4 = nn.BatchNorm2d(num_features=256)

        self.conv5 = BinarizedConv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch5 = nn.BatchNorm2d(num_features=512)

        self.conv6 = BinarizedConv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            mode=mode,
        )
        self.batch6 = nn.BatchNorm2d(num_features=512)

        self.fc1 = BinarizedLinear(1 * 512, 1024)
        self.fc2 = BinarizedLinear(1024, 1024)
        self.fc3 = BinarizedLinear(1024, classes)

        self.softmax = nn.Softmax(dim=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))  # 30
        x = self.relu(self.batch2(self.conv2(x)))  # 28
        x = self.pool(x)  # 14
        x = self.relu(self.batch3(self.conv3(x)))  # 12
        x = self.relu(self.batch4(self.conv4(x)))  # 10
        x = self.pool(x)  # 5
        x = self.relu(self.batch5(self.conv5(x)))  # 3
        x = self.relu(self.batch6(self.conv6(x)))  # 1

        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def summary(self):
        # torchsummary/torchsummary.py:48: AssertionError
        # device = device.lower()
        # assert device in [
        #    "cuda",
        #    "cpu",
        # ], "Input device is not valid, please specify 'cuda' or 'cpu'"
        # AssertionError: Input device is not valid, please specify 'cuda' or 'cpu'
        device = str(self.device).split(":")[0]
        torch_summary(
            self, input_size=(self.channels, self.height, self.width), device=device,
        )

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {
            buf.device for buf in self.buffers()
        }
        if len(devices) != 1:
            raise RuntimeError(
                "Cannot determine device: {} different devices found".format(len(devices))
            )
        return next(iter(devices))


if __name__ == "__main__":
    from omegaconf import OmegaConf

    model_conf = OmegaConf.create(
        {
            "type": "BinaryConv",
            "params": {
                "width": 32,
                "height": 32,
                "channels": 3,
                "mode": "stochastic",
                "classes": 10,
            },
        }
    )

    model_conf = OmegaConf.create(
        {
            "type": "BinaryLinear",
            "params": {
                "width": 28,
                "height": 28,
                "channels": 1,
                "mode": "stochastic",
                "classes": 10,
            },
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = BinaryConv(model_conf)
    model = BinaryLinear(model_conf)
    model = model.to(device)
    model.summary()
