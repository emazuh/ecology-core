import os
import cv2
import pickle
import PIL

import numpy as np
import pandas as pd
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data_transforms import get_default_transforms

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


class SSW60Dataset(Dataset):
    """
    Loads video dataset for the 60 class birds dataset provided 
    as part of SSW60 dataset <https://github.com/visipedia/ssw60>.
    """
    
    def __init__(self, 
                 transform: transforms.transforms.Compose = None, 
                 split: str ='train', 
                 skip_transform: bool = False, 
                 args: Namespace = None) -> None:
        """
        Creates a pytorch dataloader for SW60 dataset. If no transform is
        provided, use the default transforms (inherited from Efficientformer).

        @param transform: used to provide a custom data transform
        @param split: specifies data split `train|val|test`
        @param skip_transform: if set, perform no data transforms
        @param args: parsed command line arguments
        """
        md = pd.read_csv(os.path.join(CURRENT_PATH, f'./metadata/ssw60/{split}_video_ml.csv'))
        if args.data_subset < 1.0 and split == 'train':
            md = md.sample(frac=args.data_subset)

        self.md = md
        self.split = split
        self._args = args
        self._args.classes = 60
        self.n_vid_frames = args.n_vid_frames
        self.n_train_frames = args.n_train_frames
        self.rootdir = f'/scr/{os.environ["USER"]}_data/ssw60_video'
        
        if hasattr(args, 'train_data_folder') and args.train_data_folder is not None:
            # Overrides the data folder if needed
            self.rootdir = args.train_data_folder 
        
        # select k shot data if requested
        # TODO(@emazuh): update to support other values of k
        if args.k_shot == 5 and split in ['train', 'val']:
            ks_md = pd.read_csv(os.path.join(CURRENT_PATH, f'./metadata/ssw60/train_video_ml.csv'))
            with open(os.path.join(CURRENT_PATH, "metadata/ssw60/ssw60_split_indices.pkl"), "rb") as f:
                split_indices = pickle.load(f)[split]
            self.md = ks_md.iloc[split_indices]
            print('[INFO]: Using k-shot =', args.k_shot, 'for ssw60')
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
        idx = md_idx
        
        label = metadata.label
        asset_id_path = '/'.join(list(str(metadata.asset_id))[:4])

        # open video file with opencv
        video_path =  os.path.join(self.rootdir, f'video/{asset_id_path}/{metadata.asset_id}.mp4')
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # select a random start frame and `n_train_frames` consecutive frames during training
        offset = np.random.randint(max(1, frame_count - self.n_train_frames))
        video_frames = []
        if self.split == 'train':
            frame_numbers = [offset + frame_number for frame_number in range(self.n_train_frames)]
        else:
            # for evaluation select 8 evenly spaced frames as is done in the original ssw60 paper
            frame_numbers = [(frame_number * 2) + 2 for frame_number in range(self.n_vid_frames)]

        for frame_number in frame_numbers:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video_capture.read()
            if image is None:
                # use a frame of zeros if video is empty
                image = np.zeros((255, 255, 3))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frames.append(image)

        # close video file after reading data
        video_capture.release()
        
        if self.transform is not None:
            # transform each frame independently
            video_frames = torch.stack([self.transform(PIL.Image.fromarray(im.astype(np.uint8))) for im in video_frames]).float()

            if len(video_frames.shape) == 3:
                # single image video (extend one more axis)
                video_frames = video_frames.unsqueeze(0)

        if self._args.visualization:
            return image, label, (video_path, frame_numbers)
        return video_frames, label
        

def get_ssw60_vision_dataset(args=None):
    """
    Returns train/val/test split of the ssw60 dataset
    """

    dataset_train = SSW60Dataset(split='train', args=args)
    dataset_val = SSW60Dataset(split='val', args=args)
    dataset_test = SSW60Dataset(split='test', args=args)
    
    return dataset_train, dataset_val, dataset_test