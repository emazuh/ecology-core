import torch
import random
from torch.nn import functional as F

# -------------------------
# Video utilities
# -------------------------

def flatten_video_batch(X):
    """
    Input: [B, T, C, H, W]
    Output: (flattened_X, (B, T))
    """
    B, T, C, H, W = X.shape
    return X.reshape(B * T, C, H, W), (B, T)

def unflatten_video_logits(logits, shape):
    """ shape = (B, T) """
    B, T = shape
    return logits.reshape(B, T, -1).mean(1)

# -------------------------
# Balanced subset sampling
# -------------------------

def balanced_subset_indices(labels, samples_per_class=5):
    """
    labels: LongTensor of dataset labels
    returns indices balanced across classes
    """
    indices = []
    labels = labels.cpu()
    classes = torch.unique(labels).tolist()

    for c in classes:
        cls_idx = torch.where(labels == c)[0]
        if len(cls_idx) == 0:
            continue
        take = min(samples_per_class, len(cls_idx))
        chosen = cls_idx[torch.randperm(len(cls_idx))[:take]]
        indices.extend(chosen.tolist())

    random.shuffle(indices)
    return indices

def build_subset_loader(val_loader, samples_per_class=5):
    """
    Builds a new loader containing a balanced subset of val.
    Assumes val_loader.dataset has a 'targets' or 'labels' attribute.
    """
    dataset = val_loader.dataset
    if hasattr(dataset, "targets"):
        labels = torch.tensor(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = torch.tensor(dataset.labels)
    else:
        return val_loader  # fallback: standard val_loader

    subset_idx = balanced_subset_indices(labels, samples_per_class)

    from torch.utils.data import Subset, DataLoader
    subset = Subset(dataset, subset_idx)
    return DataLoader(
        subset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=True
    )
