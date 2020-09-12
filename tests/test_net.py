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
