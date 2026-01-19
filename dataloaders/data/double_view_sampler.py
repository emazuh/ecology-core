import copy
import math
import random
import torch
from typing import List, Iterator

class DoubleViewTrainSampler:
    """TODO: Update to double view only
    Double-view batch sampler for contrastive learning. This sampler yields batches of fixed batch size, but each has 
    images with one random transform applied to the first half of the batch and another random transform to the
    second half of the batch (same original images as the first half).

    Args:
        opts: command line argument
        n_data_samples: Number of samples in the dataset
        is_training: Training or validation mode. Default: False
    """

    def __init__(
        self,
        opts,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:

        n_gpus: int = max(1, torch.cuda.device_count())
        batch_size_gpu0: int = opts.batch_size # 64 # 128 # 256 # 128 # get_batch_size_from_opts(opts, is_training=is_training)
        self.batch_size = opts.batch_size
        
        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus
        
        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[: (total_size - n_data_samples)]
        assert total_size == len(indexes)

        self.img_indices = indexes
        self.n_samples = total_size
        self.double_view_image = opts.double_view_image
        self.shuffle = True if is_training else False
        self.is_training = is_training
        self.epoch = 0

    def get_indices(self) -> List[int]:
        """
        Returns a list of indices of dataset elements to iterate over.
        """
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(img_indices)
        return img_indices

    def __len__(self) -> int:
        return len(self.img_indices)
        
    def __iter__(self) -> Iterator[int]:
        img_indices = self.get_indices()
        start_index = 0
        n_samples = len(img_indices)
        while start_index < n_samples:

            batch_size = self.batch_size
            if self.double_view_image: batch_size //= 2
            end_index = min(start_index + batch_size, n_samples)
            batch_ids = img_indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size and self.is_training:
                batch_ids += img_indices[: (batch_size - n_batch_samples)]
            if self.double_view_image:
                batch_ids = batch_ids + batch_ids
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [b_id for b_id in batch_ids]
                yield batch