import os
import pickle
import json
import PIL

import pandas as pd
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import load_dataset

from .data_transforms import get_default_transforms

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


class RareSpeciesDataset(Dataset):
    """
    Loads image dataset for the 400 class Rare Species dataset provided 
    as part of Imageomics BioCLIP datasets.
    """

    def __init__(self,
                 ds: datasets.Dataset,
                 transform: transforms.transforms.Compose = None,
                 split: str ='train',
                 skip_transform: bool = False,
                 args: Namespace = None) -> None:
        """
        Creates a pytorch dataloader for birds_inat dataset. If no transform is
        provided, use the default transforms (inherited from Efficientformer).

        @param ds: rare_species dataset from huggingface datasets API
        @param transform: used to provide a custom data transform
        @param split: specifies data split `train|val|test`
        @param skip_transform: if set, perform no data transforms
        @param args: parsed command line arguments
        """
        
        
        # md = pd.read_csv(os.path.join(CURRENT_PATH, f'./metadata/rare_species/seen_in_training.json')
        # if args.data_subset < 1.0 and split == 'train':
        #    md = md.sample(frac=args.data_subset)
        
        # Build class mapping (species to index)
        # self.classes = sorted(set(hf_dataset[target_col]))
        # self.class2idx = {c: i for i, c in enumerate(self.classes)}

        with open(os.path.join(CURRENT_PATH, "metadata/rare_species/rare_species_split_indices.pkl"), "rb") as f:
            split_indices = pickle.load(f)

        indices = split_indices['test'] if split == 'test' else split_indices['train']
        self.split = split
        self._args = args
        self._args.classes = 400
        self.ds = ds.select(indices)
        self.ds = self.ds.cast_column("image", datasets.Image(decode=False))

        self.transform = transform
        self.skip_transform = skip_transform
        if not self.transform and not self.skip_transform:
            self.transform = get_default_transforms(split, args)


    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        img_path = item["image"]["path"]  # This is already a PIL.Image
        image = PIL.Image.open(img_path).convert('RGB')
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        if self._args.visualization:
            return image, label, img_path
        return image, label
    

# def extract_filename(example):
#     # Add actual filename field for matching
#     # This extracts filename from internal dataset format
#     if hasattr(example["file_name"], "filename"):
#         filename = os.path.basename(example["file_name"].filename)
#     elif "path" in example["file_name"].info:
#         filename = os.path.basename(example["file_name"].info["path"])
#     else:
#         # fallback or skip
#         filename = None
#     return {"filename": filename}
    
def get_rare_species_dataset(args=None):
    """
    Returns train/val/test split of the are_species dataset
    """

    # Load the seen species JSON
    # https://github.com/Imageomics/bioclip/blob/main/data/rarespecies/seen_in_training.json
    # https://github.com/Imageomics/bioclip/blob/main/data/rarespecies/unseen_in_training.json
    # https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/RareSpecies_DatasetCard.md
    with open(os.path.join(CURRENT_PATH, f'./metadata/rare_species/seen_in_training.json')) as f:
        seen_dict = json.load(f)

    # Flatten to get all seen image file names
    seen_image_filenames = set()
    for paths in seen_dict.values():
        for p in paths:
            fname = os.path.basename(p)  # Extract "10431575_...jpg"
            seen_image_filenames.add(fname)
    
    # Load the dataset
    os.environ['HF_DATASETS_CACHE'] = f'/scr/{os.environ["USER"]}_data/huggingface'
    os.environ["HF_HUB_CACHE"] = f'/scr/{os.environ["USER"]}_data/huggingface'
    os.environ["TMPDIR"] = f'/scr/{os.environ["USER"]}_data/huggingface'
    os.environ["TRANSFORMERS_CACHE"] = f'/scr/{os.environ["USER"]}_data/huggingface'
    os.environ["HF_HOME"] = f'/scr/{os.environ["USER"]}_data/huggingface'
    
    # ds = load_dataset("imageomics/rare-species", split="train", cache_dir=f'/scr/{os.environ["USER"]}_data/rare_species')
    ds = load_dataset("imagefolder", data_dir=f'/scr/{os.environ["USER"]}_data/rare_species', 
                      split="train", cache_dir=f'/scr/{os.environ["USER"]}_data/huggingface')

    # ds_with_paths = ds.cast_column("image", datasets.Image(decode=False))
    
    # Extract and add 'filepath' column
    # def add_filepath(example):
    #     path = example["image"]["path"]
    #     return {"filepath": path}
    
    # ds_with_paths = ds_with_paths.map(add_filepath)
    
    # Copy 'filepath' column into the original decoded dataset
    # ds = ds.add_column("filename", ds_with_paths["filepath"])

    # ds = load_dataset(
    #     "parquet",
    #     data_files="/gscratch/scrubbed/inaturalist/rare-species/data/*.parquet",
    #     split="train"
    # )
    # print('columns', ds[0])
    # # Add a new field to the dataset
    # ds = ds.map(extract_filename)
    
    # Filter into seen / unseen
    # seen_ds = ds.filter(lambda x: x["filename"] in seen_image_filenames)
    # unseen_ds = ds.filter(lambda x: x["filename"] not in seen_image_filenames)

    dataset_train = RareSpeciesDataset(ds=ds, split='train', args=args, skip_transform=args.double_view_image)
    dataset_val = RareSpeciesDataset(ds=ds, split='val', args=args)
    dataset_test = RareSpeciesDataset(ds=ds, split='test', args=args)

    if args.double_view_image:
        from . import DoubleViewTrainDataset
        dataset_train = DoubleViewTrainDataset(dataset_train, args=args)
    return dataset_train, dataset_val, dataset_test

