"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import sys
from argparse import ArgumentParser

import torch
from torchvision import transforms

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    return parser.parse_args()

def main(args):
    print(torch.cuda.device_count(), 'gpus available')
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="train")
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="val")
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'test'), train_transforms, split="test")

    torch.save({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, os.path.join(args.data, 'datasets.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
