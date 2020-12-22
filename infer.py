"""
Usage:
    main.py predict [options] [--config=<model config path>] [--weights-filepath=<weights file path>] [--image-path=<image path>]
    main.py predict (-h | --help)
Options:
    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/deterministic_conv/config.yaml] [type: path]
    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/deterministic_conv/weights.ckpt] [type: path]    
    --image-path <image path> Path to image filepath for inference  [default: images/cifar_frog(6).png]
            
    -h --help  Show this.
"""
import pytorch_lightning
import torch

from PIL import Image
from omegaconf import DictConfig

from src.runner.predictor import Predictor
from src.utils import get_config


pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def infer(hparams: dict):
    weight_filepath = str(hparams.get("--weights-filepath"))
    image_path = str(hparams.get("--image-path"))

    config_list = ["--config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)

    predictor = Predictor(config=config)
    predictor.load_state_dict(torch.load(weight_filepath)["state_dict"])
    predictor.eval()

    image: Image = Image.open(image_path)
    image: torch.Tensor = predictor.preprocess(image)

    predictions: str = predictor(image)
    print(f"pred: {predictions}, target: {image_path.split('_')[-1].split('.')[0]}")
