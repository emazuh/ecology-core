"""
Provides supported models for experiments:

efficientformerV2-{s0,s1,s2}
mobilevitV2-{0.5,0.75,1.0}
efficientvit-{m0,m1,m2}
efficientnetV2-{s}
efficientnet-{b0,b3}
mobilenetV2
mobilenetV3-{small}
"""

import os
import argparse
import timm
import torch
import torch.nn as nn
import torchvision.models as models

from .models_utils import get_pt_path_from_dir


def get_model(args):
    """
    Build a model based on provided model name.
    """
    
    model = None
    if 'efficientformer' in args.vision_model:
        model = build_efficientformer_model(args)
    elif 'mobilevit' in args.vision_model:
        model = build_mobilevit_model(args)
    elif 'efficientvit' in args.vision_model:
        model = build_efficientvit_model(args)
    elif 'efficientnet-v2' in args.vision_model:
        model = build_efficientnet_v2(args)
    elif 'efficientnet' in args.vision_model:
        model = build_efficientnet_model(args)
    elif 'mobilenet-v3' in args.vision_model:
        model = build_mobilenet_v3_model(args)
    elif 'mobilenet' in args.vision_model:
        model = build_mobilenet_v2_model(args)
    else:
        print('[WARNING]: Exiting.. Unrecognized model', args.vision_model)
        exit(1)

    if args.from_custom_pretrained and 'moe' not in args.vision_model: # moe loads pretrained differently
        if args.pretrained_path == 'imagenet':
            print('[INFO]: using original imagenet pretrained model weights for', args.vision_model)
        elif os.path.isdir(args.pretrained_path) and 'ssw60' in args.data_path:
            pt_path = get_pt_path_from_dir(args.pretrained_path, args.vision_model)
            print('[INFO]: Using pretrained_path', pt_path, 'for vision model', args.vision_model)
            model.load_state_dict(torch.load(pt_path)['model_state_dict'])
        else:
            print('[INFO]: Using pretrained_path', args.pretrained_path)
            model.load_state_dict(torch.load(args.pretrained_path)['model_state_dict'])
        
    model = model.to(args.device)
    return model

def build_mobilevit_model(args):
    name = 'mobilevitv2_050'
    if '0.75' in args.vision_model:
        name = 'mobilevitv2_075'
    elif '1.0' in args.vision_model:
        name = 'mobilevitv2_100'
    model = timm.create_model(name, pretrained=True, num_classes=args.classes,
                            drop_rate=args.dropout_rate)
    # if args.from_custom_pretrained and args.pretrained_path is not None: 
    #     # moe loads pretrained before creating experts
    #     if args.pretrained_path == 'imagenet':
    #         print('[INFO]: using original imagenet pretrained model weights for', args.vision_model)
    #     elif os.path.isdir(args.pretrained_path) and 'ssw60' in args.data_path:
    #         pt_path = get_pt_path_from_dir(args.pretrained_path, args.vision_model)
    #         print('[INFO]: Using pretrained_path', pt_path, 'for vision model', args.vision_model)
    #         model.load_state_dict(torch.load(pt_path)['model_state_dict'])
    #     else:
    #         print('[INFO]: Using pretrained_path', args.pretrained_path)
    #         model.load_state_dict(torch.load(args.pretrained_path)['model_state_dict'])
    return model

def build_efficientformer_model(args):
    name = 'efficientformerv2_s0.snap_dist_in1k'
    if 's1' in args.vision_model:
        name = 'efficientformerv2_s1.snap_dist_in1k'
    elif 's2' in args.vision_model:
        name = 'efficientformerv2_s2.snap_dist_in1k'

    model = timm.create_model(name, pretrained=True, num_classes=args.classes,
                            drop_rate=args.dropout_rate)
    # if args.from_custom_pretrained and args.pretrained_path is not None: 
    #     # moe loads pretrained before creating experts
    #     if args.pretrained_path == 'imagenet':
    #         print('[INFO]: using original imagenet pretrained model weights for', args.vision_model)
    #     elif os.path.isdir(args.pretrained_path) and 'ssw60' in args.data_path:
    #         pt_path = get_pt_path_from_dir(args.pretrained_path, args.vision_model)
    #         print('[INFO]: Using pretrained_path', pt_path, 'for vision model', args.vision_model)
    #         model.load_state_dict(torch.load(pt_path)['model_state_dict'])
    #     else:
    #         print('[INFO]: Using pretrained_path', args.pretrained_path)
    #         model.load_state_dict(torch.load(args.pretrained_path)['model_state_dict'])
    return model

def build_efficientvit_model(args):
    name = 'efficientvit_m0'
    if 'm1' in args.vision_model:
        name = 'efficientvit_m1'
    elif 'm2' in args.vision_model:
        name = 'efficientvit_m2'
    elif 'm4' in args.vision_model:
        name = 'efficientvit_m4'
    return timm.create_model(name, pretrained=True, num_classes=args.classes,
                            drop_rate=args.dropout_rate)


def build_efficientnet_v2(args):
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[0] = nn.Dropout(p=args.dropout_rate, inplace=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=args.classes)
    return model


def build_efficientnet_model(args):
    if 'b0' in args.vision_model:
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=args.classes)
    elif 'b3' in args.vision_model:
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(in_features=1536, out_features=args.classes)
    else:
        print(f'Model {args.vision_model} is not recognized.')
        exit()

    model.classifier[0] = nn.Dropout(p=args.dropout_rate, inplace=False)

    return model


def build_mobilenet_v2_model(args):
    model = models.mobilenet_v2(pretrained=True)
    
    model.classifier[0] = nn.Dropout(p=args.dropout_rate, inplace=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=args.classes)

    return model


def build_mobilenet_v3_model(args):
    
    if 'large' in args.vision_model:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        
        model.classifier[2] = nn.Dropout(p=args.dropout_rate, inplace=False)
        model.classifier[3] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=args.classes)
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        model.classifier[2] = nn.Dropout(p=args.dropout_rate, inplace=False)
        model.classifier[3] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=args.classes)

    return model