import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf

from src.model.net import BinaryConv, BinaryLinear


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module")
def tearup_binarylinear_model_config():
    return OmegaConf.create(
        {
            "params": {
                "width": 28,
                "height": 28,
                "channels": 1,
                "in_feature": 784,
                "classes": 10,
                "mode": "stochastic",
                "feature_layers": {
                    "linear": [
                        {
                            "in_feature": 784,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 10,
                            "bias": True,
                            "batch_norm": True,
                            "activation": None,
                            "mode": "stochastic",
                        },
                    ]
                },
                "output_layer": {"type": "Softmax", "args": {"dim": 1}},
            },
            "type": "BinaryLinear",
        }
    )


binarylinear_forward_test_case = [
    # (device, test_input)
    ("cpu", torch.randn(((2, 1, 28, 28)))),
    (torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.randn(((2, 1, 28, 28)))),
]


@pytest.mark.parametrize(
    "device, test_input", binarylinear_forward_test_case,
)
def test_binarylinear_forward(
    fix_seed, tearup_binarylinear_model_config, device, test_input,
):

    model = BinaryLinear(tearup_binarylinear_model_config).to(device)

    test_input = test_input.to(device)
    model(test_input)


@pytest.fixture(scope="module")
def tearup_binaryconv_model_config():
    return OmegaConf.create(
        {
            "params": {
                "width": 32,
                "height": 32,
                "channels": 3,
                "classes": 10,
                "mode": "stochastic",
                "feature_layers": {
                    "conv": [
                        {
                            "in_channels": 3,
                            "out_channels": 128,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 128,
                            "out_channels": 128,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": {
                                "type": "MaxPool2d",
                                "args": {
                                    "kernel_size": [2, 2],
                                    "stride": None,
                                    "padding": 0,
                                    "dilation": 1,
                                    "return_indices": False,
                                    "ceil_mode": False,
                                },
                            },
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 128,
                            "out_channels": 256,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 256,
                            "out_channels": 256,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": {
                                "type": "MaxPool2d",
                                "args": {
                                    "kernel_size": [2, 2],
                                    "stride": None,
                                    "padding": 0,
                                    "dilation": 1,
                                    "return_indices": False,
                                    "ceil_mode": False,
                                },
                            },
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 256,
                            "out_channels": 512,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_channels": 512,
                            "out_channels": 512,
                            "kernel_size": 3,
                            "stride": 1,
                            "padding": 0,
                            "dilation": 1,
                            "groups": 1,
                            "bias": True,
                            "padding_mode": "zeros",
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "pool": None,
                            "mode": "stochastic",
                        },
                    ],
                    "linear": [
                        {
                            "in_feature": 512,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": None,
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 1024,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                        {
                            "in_feature": 1024,
                            "out_feature": 10,
                            "bias": True,
                            "batch_norm": True,
                            "activation": {"type": "ReLU", "args": {}},
                            "mode": "stochastic",
                        },
                    ],
                },
                "output_layer": {"type": "Softmax", "args": {"dim": 1}},
            },
            "type": "BinaryConv",
        }
    )


binaryconv_forward_test_case = [
    # (device, test_input)
    ("cpu", torch.randn(((2, 3, 32, 32)))),
    (torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.randn(((2, 3, 32, 32)))),
]


@pytest.mark.parametrize(
    "device, test_input", binaryconv_forward_test_case,
)
def test_binaryconv_forward(
    fix_seed, tearup_binaryconv_model_config, device, test_input,
):

    model = BinaryConv(tearup_binaryconv_model_config).to(device)

    test_input = test_input.to(device)
    model(test_input)


summary_test_case = [
    # (device, test_input)
    ("cpu"),
    (torch.device("cuda" if torch.cuda.is_available() else "cpu")),
]


@pytest.mark.parametrize("device", summary_test_case)
def test_binarylinear_summary(fix_seed, tearup_binarylinear_model_config, device):
    model = BinaryLinear(tearup_binarylinear_model_config).to(device=device)
    model.summary()


@pytest.mark.parametrize("device", summary_test_case)
def test_binaryconv_summary(fix_seed, tearup_binaryconv_model_config, device):
    model = BinaryConv(tearup_binaryconv_model_config).to(device=device)
    model.summary()
