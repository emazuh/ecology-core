import os
import pickle
import PIL

import pandas as pd
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data_transforms import get_default_transforms

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


class BirdsInatDataset(Dataset):
    """
    Loads image dataset for the 60 class iNat '21 birds dataset provided 
    as part of SSW60 dataset <https://github.com/visipedia/ssw60>.
    """
    
    def __init__(self, 
                 transform: transforms.transforms.Compose = None, 
                 split: str ='train', 
                 skip_transform: bool = False, 
                 args: Namespace = None) -> None:
        """
        Creates a pytorch dataloader for birds_inat dataset. If no transform is
        provided, use the default transforms (inherited from Efficientformer).

        @param transform: used to provide a custom data transform
        @param split: specifies data split `train|val|test`
        @param skip_transform: if set, perform no data transforms
        @param args: parsed command line arguments
        """
        md = pd.read_csv(os.path.join(CURRENT_PATH, f'./metadata/birds_inat_{split}.csv'))
        if args.data_subset < 1.0 and split == 'train':
            md = md.sample(frac=args.data_subset)

        self.md = md
        self.split = split
        self._args = args
        self._args.classes = 60
        self.rootdir = f'/scr/{os.environ["USER"]}_data/birdsinat'
        
        if hasattr(args, 'train_data_folder') and args.train_data_folder is not None:
            # Overrides the data folder if needed
            self.rootdir = args.train_data_folder 

        # select k shot data if requested
        # if args.k_shot > 0 and split in ['train', 'val']:
        #     with open(os.path.join(CURRENT_PATH, "metadata/birds_inat/birds_inat_split_indices.pkl"), "rb") as f:
        #         split_indices = pickle.load(f)[split]
        #     self.md = self.md.iloc[split_indices]
        #     print('[INFO]: Using k-shot =', k, 'for birds_inat')
        
        # select k shot data if requested
        # TODO(@emazuh): update to support other values of k
        if args.k_shot == 5 and split in ['train', 'val']:
            ks_md = pd.read_csv(os.path.join(CURRENT_PATH, f'./metadata/birds_inat_train.csv'))
            with open(os.path.join(CURRENT_PATH, "metadata/birds_inat/birds_inat_split_indices.pkl"), "rb") as f:
                split_indices = pickle.load(f)[split]
            self.md = ks_md.iloc[split_indices]
            print('[INFO]: Using k-shot =', args.k_shot, 'for birds_inat')
            # print(self.md.head(n=20))
            
        self.transform = transform
        self.skip_transform = skip_transform
        if not self.transform and not self.skip_transform:
            self.transform = get_default_transforms(split, args)
        
    def __len__(self) -> int:
        return len(self.md)

    def __getitem__(self, idx):
        metadata = self.md.iloc[idx]
        md_idx = self.md.index[idx] # the original order of metadata in case of data subset
        label = metadata.label

        img_path = os.path.join(f'{self.rootdir}/images_inat/{metadata.img_asset_id}.jpg')
        image = PIL.Image.open(img_path) 
        
        if self.transform is not None:
            image = self.transform(image)

        if self._args.visualization:
            return image, label, img_path
        return image, label
        

def get_birds_inat_vision_dataset(args=None):
    """
    Returns train/val/test split of the birds_inat dataset
    """

    dataset_train = BirdsInatDataset(split='train', args=args, skip_transform=args.double_view_image)
    dataset_val = BirdsInatDataset(split='val', args=args)
    dataset_test = BirdsInatDataset(split='test', args=args)

    if args.double_view_image:
        from . import DoubleViewTrainDataset
        dataset_train = DoubleViewTrainDataset(dataset_train, args=args)
    return dataset_train, dataset_val, dataset_test