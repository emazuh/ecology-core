import os
import sys
import pickle
import pandas as pd
import json
import numpy as np
import pathlib
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset

from .data_transforms import get_default_transforms

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
USER = os.environ['USER']


def get_mask_non_empty(dataset):
    metadf = pd.read_csv(dataset._data_dir / 'metadata.csv')
    filename = os.path.expanduser(dataset._data_dir / 'iwildcam2020_megadetector_results.json')
    with open(filename, 'r') as f:
        md_data = json.load(f)
    id_to_maxdet = {x['id']: x['max_detection_conf'] for x in md_data['images']}
    threshold = 0.95
    mask_non_empty = [id_to_maxdet[x] >= threshold for x in metadf['image_id']]
    return mask_non_empty


def get_nonempty_subset(dataset, split, frac=1.0, transform=None):
    if split not in dataset.split_dict:
        raise ValueError(f"Split {split} not found in dataset's split_dict.")
    split_mask = dataset.split_array == dataset.split_dict[split]

    # intersect split mask with non_empty. here is the only place this fn differs
    # from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/wilds_dataset.py#L56
    mask_non_empty = get_mask_non_empty(dataset)
    split_mask = split_mask & mask_non_empty

    split_idx = np.where(split_mask)[0]
    if frac < 1.0:
        num_to_retain = int(np.round(float(len(split_idx)) * frac))
        split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
    subset = WILDSSubset(dataset, split_idx, transform)
    return subset


class IWildCam:
    """
    Loads iWildCam dataset used in the camera trap distribution 
    shift challenge https://wilds.stanford.edu/leaderboard/#iwildcam
    """
    
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 remove_non_empty=False,
                 batch_size=128,
                 num_workers=16,
                 classnames=None,
                 subset='train'):
        # https://github.com/p-lambda/wilds/blob/main/wilds/datasets/archive/iwildcam_v1_0_dataset.py#L16
        self.dataset = wilds.get_dataset(dataset='iwildcam', root_dir=location)
        self.train_dataset = self.dataset.get_subset('train', transform=preprocess)
        self.train_loader = get_train_loader("standard", self.train_dataset, num_workers=num_workers, batch_size=batch_size)

        if remove_non_empty:
            self.train_dataset = get_nonempty_subset(self.dataset, 'train', transform=preprocess)
        else:
            self.train_dataset = self.dataset.get_subset('train', transform=preprocess)

        if remove_non_empty:
            self.test_dataset = get_nonempty_subset(self.dataset, subset, transform=preprocess)
        else:
            self.test_dataset = self.dataset.get_subset(subset, transform=preprocess)

        self.test_loader = get_eval_loader(
            "standard", self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size)

        labels_csv = os.path.join(CURRENT_PATH, './metadata/iwildcam_metadata/labels.csv')
        # labels_csv = '../metadata/iwildcam_metadata/labels.csv'
        df = pd.read_csv(labels_csv)
        df = df[df['y'] < 99999]
        
        self.classnames = [s.lower() for s in list(df['english'])]
        
class IWildCamIDVal(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_val'
        super().__init__(*args, **kwargs)

class IWildCamID(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)

class IWildCamOOD(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)


class IWildCamNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'train'
        super().__init__(*args, **kwargs)


class IWildCamIDNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'id_test'
        super().__init__(*args, **kwargs)


class IWildCamOODNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'test'
        super().__init__(*args, **kwargs)
        
class IWildCamValNonEmpty(IWildCam):
    def __init__(self, *args, **kwargs):
        kwargs['subset'] = 'val'
        super().__init__(*args, **kwargs)

class IWildCamDataset(Dataset):
    """
    Loads iWildCam dataset used in the camera trap distribution 
    shift challenge (supporting in distribution and out of 
    distribution datasets for evaluation).
    """
    
    def __init__(self, 
                 transform: transforms.transforms.Compose = None, 
                 split: str ='train', 
                 skip_transform: bool =False, 
                 args: Namespace = None) -> None:
        """
        Creates a pytorch dataloader for iwildcam dataset. If no transform is
        provided, use the default transforms (inherited from Efficientformer).

        @param transform: used to provide a custom data transform
        @param split: specifies data split `train|val|test`
        @param skip_transform: if set, perform no data transforms
        @param args: parsed command line arguments
        """
        self.transform = transform
        self._args = args
        self._args.classes = 182
        if not self.transform and not skip_transform:
            self.transform = get_default_transforms(split, args)

        # https://github.com/locuslab/FLYP/blob/215d5bb6feeda6675f60e5818abcb4f6465c83af/README.md?plain=1#L45
        # IWildCamIDVal,IWildCamID,IWildCamOOD
        USER = os.environ['USER']
        IWILD_LOCATION = f'/scr/{USER}_data/'

        if split == 'train':
            self.iwild_data = IWildCamNonEmpty(self.transform, location=IWILD_LOCATION).train_dataset
        elif split == 'val':
            self.iwild_data = IWildCamIDVal(self.transform, location=IWILD_LOCATION).test_dataset
        else:
            if 'ood' in args.data_path:
                print('[INFO]: Testing with iwildcam out of distribution data')
                # for testing out of distribution performance
                self.iwild_data = IWildCamOOD(self.transform, location=IWILD_LOCATION).test_dataset
            else:
                print('[INFO]: Testing with iwildcam in distribution data')
                # for testing in distribution performance
                self.iwild_data = IWildCamID(self.transform, location=IWILD_LOCATION).test_dataset

        self.indices = np.arange(len(self.iwild_data))
        if args.data_subset < 1.0 and split == 'train':
            self.indices = np.random.choice(self.indices, int(len(self.iwild_data) * args.data_subset))

        # select k shot data if requested
        # TODO(@emazuh): update to support other values of k
        if args.k_shot == 5 and split in ['train', 'val']:
            self.iwild_data = IWildCamNonEmpty(self.transform, location=IWILD_LOCATION).train_dataset
            with open(os.path.join(CURRENT_PATH, "metadata/iwildcam/iwildcam_split_indices.pkl"), "rb") as f:
                split_indices = pickle.load(f)[split]
            self.indices = np.array(split_indices)
            print('[INFO]: Using k-shot =', args.k_shot, 'for iwildcam')
            # print(self.indices[:20])
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx] # get original index in dataset for subset data
        img, label, md = self.iwild_data[idx]
        if self._args.visualization:
            image_path = str(self.iwild_data.dataset._data_dir / 'train' / self.iwild_data.dataset._input_array[idx])
            return img, label, image_path
        return img, label

def get_iwildcam_dataset(args):
    """
    Returns train/val/test split of the iwildcam dataset
    """

    dataset_train = IWildCamDataset(split='train', args=args)
    dataset_val = IWildCamDataset(split='val', args=args)
    dataset_test = IWildCamDataset(split='test', args=args)

    if args.evaluate or args.test:
        original_iwild = args.data_path
        args.data_path = 'data/iwildcam-ood'
        # Also evaluate out of distribution accuracy for iwildcam
        args.iw_ood_test_ds = IWildCamDataset(split='test', args=args)
        args.data_path = original_iwild
    return dataset_train, dataset_val, dataset_test
