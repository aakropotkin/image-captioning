import os.path
import utils
import torch
from pathlib import Path
import sys

from models.blip import blip_decoder
from tqdm import tqdm
import argparse
import numpy as np


def init_parser(**parser_kwargs):
    """
    This function initializes the parser and adds arguments to it
    :return: The parser object is being returned.
    """
    parser = argparse.ArgumentParser(description="Image caption CLI")
    parser.add_argument(
      "input",
      help="Input path, such as ./images.png or ./imgs-dir/"
    )
    parser.add_argument(
      "-e",
      "--ext",
      type=str,
      default='caption',
      help='file extension to use when saving captions'
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        default=0,
        help="gpu device to use (default=None) can be 0,1,2 for multi-gpu",
    )

    return parser


def init_model():
    """
    > Loads the model from the checkpoint file and sets it to eval mode
    :return: The model is being returned.
    """

    print("Checkpoint loading...", file=sys.stderr)
    model_path = Path( __file__ ).parent.resolve() / 'checkpoints' / 'model_large_caption.pth'
    model = blip_decoder(
        pretrained=model_path,
        image_size=384,
        vit="large"
    )
    model.eval()
    model = model.to(device)
    print(f"\nModel to {device}", file=sys.stderr)
    return model


if __name__ == "__main__":
    parser = init_parser()
    opt = parser.parse_args()

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}', file=sys.stderr)

    model_path = Path( __file__ ).parent.resolve() / 'checkpoints' / 'model_large_caption.pth'

    if not model_path.parent.is_dir():
        print(f"checkpoint directory not found.")
        utils.create_dir(model_path.parent)

    if not model_path.is_file():
        utils.download_checkpoint()

    model = init_model()
    with torch.no_grad():
        print("Inference started", file=sys.stderr)
        pil_images = utils.read_with_pil([opt.input])
        transformed_images = utils.prep_images(pil_images, device)
        image = transformed_images[0]
        caption = model.generate(
          image, sample=False, num_beams=3, max_length=77, min_length=5
        )
        print(caption[0])
