from torch.utils.data import DataLoader, TensorDataset

import sys
from .data import get_dataset

def get_default_args():
    from argparse import Namespace
    args = Namespace()
    
    args.n_vid_frames = 8
    args.n_train_frames = 2
    args.visualization = False
    args.double_view_image = False
    args.k_shot = -1
    args.evaluate = args.test = False
    args.log_wandb = True
    args.val_samples_per_class = 5
    args.max_grad_norm = 10.0
    args.eval_transform = False # True
    args.num_workers = 6
    args.from_custom_pretrained = False
    args.apply_entropy_to_val = True
    args.ds_grad_norm_file = None
    return args
    
def get_subset_dataloaders(dataset_name="data/birds_inat", batch_size=32, model='mobilevit', subset_fraction=0.1,
                          randerase_p=0.0):

    args = get_default_args()
    
    args.data_path = dataset_name
    args.data_subset = subset_fraction
    args.vision_model = model
    args.randerase_p = randerase_p
    args.batch_size = batch_size

    args.vision_model = 'efficientformer' # fix data loader for all models
    train_dataset, val_dataset, test_dataset = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    args.test_loader = test_loader
    args.vision_model = model
    return train_loader, val_loader, args

def get_dataloaders_for(dataset_name, subset_fraction=0.1, randerase_p=0.0, batch_size=32, model='mobilevit'):
    if dataset_name == "birds_inat":
        train_loader, test_loader, args = get_subset_dataloaders(subset_fraction=subset_fraction, dataset_name="data/birds_inat",
                                                                randerase_p=randerase_p, batch_size=batch_size, model=model)
        args.num_classes = 60
        print('args', args)
    elif dataset_name == "ssw60":
        train_loader, test_loader, args = get_subset_dataloaders(subset_fraction=subset_fraction, dataset_name="data/ssw60",
                                                                randerase_p=randerase_p, batch_size=batch_size, model=model)
        args.num_classes = 60
    elif dataset_name == "rare_species":
        train_loader, test_loader, args = get_subset_dataloaders(subset_fraction=subset_fraction, dataset_name="data/rare_species",
                                                                randerase_p=randerase_p, batch_size=batch_size, model=model)
        args.num_classes = 400
    elif dataset_name == "iwildcam":
        train_loader, test_loader, args = get_subset_dataloaders(subset_fraction=subset_fraction, dataset_name="data/iwildcam",
                                                                randerase_p=randerase_p, batch_size=batch_size, model=model)
        args.num_classes = 182
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_loader, test_loader, args