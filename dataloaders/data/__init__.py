import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from .birds_inat import get_birds_inat_vision_dataset
from .iwildcam import get_iwildcam_dataset
from .ssw60 import get_ssw60_vision_dataset
from .rare_species import get_rare_species_dataset

from .data_transforms import get_default_transforms

def get_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size=64, train_sampler=None, 
val_sampler=None, test_sampler=None, num_workers=4, args=None, device=None):
    """
    Wrap datasets in pytorch data loaders for the train/evalution loop.
    """
    # double view sampler takes care of variable batch size so pass bs=1 for it
    train_bs = 1 if args.double_view_image else batch_size

    train_loader = DataLoader(
        dataset_train, batch_size=train_bs, shuffle=train_sampler==None,
        num_workers=num_workers, sampler=train_sampler, pin_memory=True, drop_last=False
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=val_sampler, pin_memory=True, drop_last=False
    )

    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=test_sampler, pin_memory=True, drop_last=False
    )

    if hasattr(args, 'iw_ood_test_ds'):
        args.iw_ood_test_loader = DataLoader(args.iw_ood_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, sampler=test_sampler, pin_memory=True, drop_last=False)
    return train_loader, valid_loader, test_loader

def get_dataset(args):
    """
    Load supported dataset for training/evaluation.
    """
    if args.data_path.endswith('birds_inat'):
        dataset_train, dataset_valid, dataset_test = get_birds_inat_vision_dataset(args=args)
    elif 'iwildcam' in args.data_path:
        dataset_train, dataset_valid, dataset_test = get_iwildcam_dataset(args=args)
    elif 'rare_species' in args.data_path:
        dataset_train, dataset_valid, dataset_test = get_rare_species_dataset(args)
    elif args.data_path.endswith('ssw60'):
        dataset_train, dataset_valid, dataset_test = get_ssw60_vision_dataset(args=args)
    else:
        print('[WARNING]: Exiting.. Unsupported dataset', args.data_path)
        exit()
    print('[INFO]: Using dataset', args.data_path)

    return dataset_train, dataset_valid, dataset_test

class DoubleViewTrainDataset(Dataset):
    def __init__(self, dataset, transform=None, size=320, crop=320, double_view_image=False, randerase_p=0.5, args=None):
        self.dataset = dataset
        self.double_view_image = double_view_image
        self.transform = get_default_transforms('train', args)
    
    def __len__(self):
        if self.double_view_image: return 2*len(self.dataset)
        return len(self.dataset)

    def __getitem__(self, idxs):

        imgs = []
        labels = []
        imagenet_multiscale_transform = None
        for idx in idxs:
            image, label = self.dataset[idx]
            imgs.append(self.transform(image).unsqueeze(0))

            labels.append(label)
        return torch.cat(imgs), torch.tensor(labels).reshape(-1)
                
                
def dataset_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add command arguments for datasets
    """

    parser.add_argument(
        '--data-path', type=str, dest='data_path',
        default='data/birds_inat',
        help='Image data to use for training.'
    )
    parser.add_argument(
        '-ds', '--data-subset', type=float,
        dest='data_subset', default=1.0,
        help='Subset of entire data points to use in the experiment.'
    )
    parser.add_argument(
        '-ks', '--k-shot', type=int,
        dest='k_shot', default=-1,
        help='If positive, only use `k_shot` examples per class for train/val.'
    )
    parser.add_argument(
        '-bs', '--batch-size', type=int,
        dest='batch_size', default=32,
        help='Batch size for training the model'
    )
    parser.add_argument(
        '--classes', type=int,
        dest='classes', default=60,
        help='Number of classes for this dataset.'
    )
    parser.add_argument(
        '-dvi', '--double-view-image', action='store_true',
        dest='double_view_image',
        help='If true, use two augmented views of input image, e.g. for contrastive learning.'
    )
    parser.add_argument(
        '-mxa', '--mixup-alpha', type=float,
        dest='mixup_alpha', default=0.0,
        help='Alpha to use for mixing up images in a batch.'
    )
    parser.add_argument(
        '-edr', '--easy-ds-ratio', type=float,
        dest='easy_ds_ratio', default=0.0,
        help='Fraction of total steps to start using simple transforms during moe (from deit to only flip and randerase).'
    )
    parser.add_argument(
        '-nw', '--num-workers', type=int, default=12,
        help='Number of workers to use for the dataloader'
    )
    parser.add_argument(
        '--multiscale-sampling', action='store_true', dest='multiscale_sampling',
        help='Set this flag to use variable batch size and image crop size 160 to 320 during training.'
    )
    parser.add_argument(
        '--randerase-p',  type=float,
        dest='randerase_p', default=0.0,
        help='If greater than zero, perform torch random erase on training image with this probability.'
    )
    parser.add_argument(
        '-ntf', '--n-train-frames', type=int,
        dest='n_train_frames', default=2,
        help='Number of video frames to sample at (pseudo) random during training'
    )
    parser.add_argument(
        '-nvf', '--n-vid-frames', type=int,
        dest='n_vid_frames', default=8,
        help='Number of video frames considered per video (for validation/training frames should not be bigger than this to reduce overfitting'
    )

    return parser