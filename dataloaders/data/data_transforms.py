"""
Provides data augmentation for training vision models
based on Efficientformer default augmentation 
<https://github.com/snap-research/EfficientFormer/blob/main/util/datasets.py#L88>
which inherits from DeiT <https://github.com/facebookresearch/deit/blob/main/main.py>
"""
import timm
from torchvision import transforms
from argparse import Namespace

from torchvision import transforms

transform_moe_friendly = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0., 0., 0.], [1., 1., 1.]),
    transforms.RandomErasing(
        p=0.02, # 0.2
        scale=(0.02, 0.2), 
        ratio=(0.3, 3.3), 
        value='random'
    )
])

def get_default_transforms(split: str, args: Namespace) -> transforms.transforms.Compose:
    """
    Get data transform based on train/eval `split` and supported models.
    For `mobilevit` models, use image normalization of 0 mean and 1 std.
    Otherwise, use imagenet default image normalization.
    """

    # if 'moe' in args.vision_model and split == 'train':
    #     # print('transforms', transform_moe_friendly)
    #     # return transform_moe_friendly
    MEAN = [0., 0., 0.] if 'mobilevit' in args.vision_model else [0.485, 0.456, 0.406]
    STD = [1., 1., 1.] if 'mobilevit' in args.vision_model else [0.229, 0.224, 0.225]
    transform_simple = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(
            p=args.randerase_p, # 0.2
            scale=(0.02, 0.2), 
            ratio=(0.3, 3.3), 
            value='random'
        )
    ])
    args.transform_simple = transform_simple

    # Use the same augmentation as efficientformer
    model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    data_config['re_prob'] = 0.0 if not hasattr(args, 'randerase_p') else args.randerase_p
    data_config['re_mode'] = 'pixel'

    if 'mobilevit' in args.vision_model:
        # mobilevit does not use imagenet mean/std statistics
        data_config['mean'], data_config['std'] = [0., 0., 0.], [1., 1., 1.]
        
    if hasattr(args, 'eval_transform') and args.eval_transform: 
        # force validation transform regardless of split. Useful for example for generating
        # clustering embeddings from train data
        print(f'[INFO]: Overriding split {split} transform with eval split transform')
        split = 'val' 
    return timm.data.create_transform(**data_config, is_training=split=='train')

