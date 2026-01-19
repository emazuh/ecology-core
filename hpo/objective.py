import wandb
import timm
import optuna
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.search_space import suggest_stage1_hyperparameters, suggest_stage2_hyperparameters
from configs.adapter_config import AdapterConfig
from adapters.injector import inject_adapters_eformer, inject_adapters_for_model
from adapters.freeze import freeze_all_except
from training.engine import train_and_eval
from dataloaders.dataloaders import get_dataloaders_for
from utilities.reproducibility import set_global_seed

import sys
import os
current_directory = os.getcwd()
sys.path.insert(0, os.path.join(current_directory, '../'))
from models import get_model

    
def objective_general(trial, seed=42, epochs=20, data_subset=0.1, dataset_name="birds_inat", model_name="mobilevit",
                      log_wandb=True):
    set_global_seed(seed)

    hp = suggest_stage1_hyperparameters(trial, epochs)

    lr = hp["lr"]
    batch_size = hp["batch_size"]
    randerase_p = hp["randerase_p"]
    dropout = hp["dropout"]
    weight_decay = hp["weight_decay"]
    
    # Load small subset of dataset
    train_loader, val_loader, args = get_dataloaders_for(dataset_name, subset_fraction=data_subset, randerase_p=randerase_p,
                                                        batch_size=batch_size, model=model_name)
    # train_loader, val_loader, args = get_subset_dataloaders(subset_fraction=data_subset, dataset_name=dataset_name,
    #                                                        randerase_p=randerase_p)
    # args.num_classes = NUM_CLASSES
    args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    args.batch_size = batch_size
    args.dropout_rate = dropout
    args.lr = lr
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.log_wandb = log_wandb
    args.ds_grad_norm_file = None
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if log_wandb:
        wandb.init(
            project="mobilevit-adapter-ablation",
            reinit=True,
            config={
                "lr": round(lr, 4),
                "dropout": round(dropout, 2),
                "randerase_p": round(randerase_p, 2),
                "batch_size": batch_size,
                "data_subset": data_subset,
                "dataset_name": dataset_name,
                "weight_decay": weight_decay,
            },
            tags=["cvpr25optuna_birdsmvit", "general_hyperparams", f"{dataset_name}_{100*data_subset}p"]
        )
    # Model setup
    # model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=args.num_classes, drop_rate=dropout)
    model = get_model(args)
    model.to(args.device)
    
    
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    val_acc = train_and_eval(model, train_loader, val_loader, args=args, run_test=False)
    return val_acc

def objective_mlp(trial, adapter_cfg, seed=42, epochs=20, data_subset=0.1, dataset_name="birds_inat", model_name="mobilevit",
                  log_wandb=True, chain_type="par_fixed", ds_grad_norm_file='birds_grad_norm_per_layer.csv'):
    set_global_seed(seed)

    # hp = suggest_stage1_hyperparameters(trial, epochs)
    # from rare species
    hp = {'lr': 0.009970171133305596, 'batch_size': 64, 'dropout': 0.012265601706912002, 
          'randerase_p': 0.25825040732523524, 'weight_decay': 4.672701320911668e-05, 
          'warmup_ratio': 0.00036728817232678165}

    lr = hp["lr"]

    # TODO: Search over these in stage 2
    hp["chain_type"] = chain_type
    hp["B5_adapters"] = 1
    hp["B6_adapters"] = 1
    hp["freeze_backbone"] = False # True
    
    mlp_reduction = trial.suggest_categorical("mlp_reduction", [1, 2, 4, 8, 16, 32, 64])
    hp["mlp_reduction"] = mlp_reduction
    
    batch_size = hp["batch_size"]
    if ds_grad_norm_file: 
        batch_size = 16
    # else:
    #     batch_size = 64
    randerase_p = hp["randerase_p"]
    dropout = hp["dropout"]
    weight_decay = hp["weight_decay"]
    mlp_reduction = hp["mlp_reduction"]
    
    # Load small subset of dataset
    train_loader, val_loader, args = get_dataloaders_for(dataset_name, subset_fraction=data_subset, randerase_p=randerase_p,
                                                        batch_size=batch_size, model=model_name)
    # train_loader, val_loader, args = get_subset_dataloaders(subset_fraction=data_subset, dataset_name=dataset_name,
    #                                                        randerase_p=randerase_p)
    # args.num_classes = NUM_CLASSES
    args.warmup_ratio = hp["warmup_ratio"] # trial.suggest_float("warmup_ratio", 0.0, 0.1)
    args.batch_size = batch_size
    args.dropout_rate = dropout
    args.lr = lr
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.log_wandb = log_wandb
    args.mlp_reduction = mlp_reduction
    
    args.ds_grad_norm_file = ds_grad_norm_file
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if log_wandb:
        wandb.init(
            project="mobilevit-adapter-ablation",
            reinit=True,
            config={
                "lr": round(lr, 4),
                "dropout": round(dropout, 2),
                "randerase_p": round(randerase_p, 2),
                "batch_size": batch_size,
                "data_subset": data_subset,
                "dataset_name": dataset_name,
                "weight_decay": weight_decay,
                "mlp_reduction": mlp_reduction,
            },
            tags=["cvpr25optuna_birdsmvit", "backbone_mlp_reduction2", f"{dataset_name}_{100*data_subset}p"]
        )
    # Model setup
    # model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=args.num_classes, drop_rate=dropout)
    model = get_model(args)
    unfreeze_last2 = ["stages.3", "stages.4", "head", "adapter"]
    model = inject_adapters_for_model(model, model_name, adapter_cfg, unfreeze_last2, hp, args)
    print("total_params ", sum(p.numel() for p in model.parameters()))
    model.to(args.device)
    
    
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    val_acc = train_and_eval(model, train_loader, val_loader, args=args, run_test=False)
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params", total_params)
    return val_acc, total_params

def objective_adapter_layers(trial, adapter_cfg, seed=42, epochs=20, data_subset=0.1, dataset_name="birds_inat", model_name="mobilevit",
                  log_wandb=True, chain_type="par_fixed", ds_grad_norm_file='birds_grad_norm_per_layer.csv', single_objective=False):
    set_global_seed(seed)

    # hp = suggest_stage1_hyperparameters(trial, epochs)
    # from rare species
    hp = {'lr': 0.009970171133305596, 'batch_size': 64, 'dropout': 0.012265601706912002, 
          'randerase_p': 0.25825040732523524, 'weight_decay': 4.672701320911668e-05, 
          'warmup_ratio': 0.00036728817232678165}

    lr = hp["lr"]
    hp["place_on"], hp["layer_mode"] = None, None
    if adapter_cfg is None:
        hp["place_on"] = trial.suggest_categorical("place_on", ["mlp", "conv"])
        hp["layer_mode"] = trial.suggest_categorical("layer_mode", ["every", "all"])

    hp["adapter_type"] = trial.suggest_categorical("adapter_type", ["simple", "fourier", "bottleneck"]) #  
    adapter_type = hp["adapter_type"]
    entropy_coeff = hp["entropy_coeff"] = trial.suggest_categorical("entropy_coeff", [ 0.1,  1e-2, 1e-3]) # # -1e-2, , -1e-3 -0.1, 
    
    # TODO: Search over these in stage 2
    hp["chain_type"] = chain_type
    # hp["B5_adapters"] = 1
    # hp["B6_adapters"] = 1
    
    # Adapter fan-out
    hp["B5_adapters"] = trial.suggest_categorical("B5_adapters", [1, 2]) # 4, 8
    hp["B6_adapters"] = trial.suggest_categorical("B6_adapters", [1, 2, 4, 8])
    freeze_backbone = hp["freeze_backbone"] = trial.suggest_categorical("freeze_backbone", [False, True]) # False # True
    
    # mlp_reduction = trial.suggest_categorical("mlp_reduction", [2, 4, 8, 16, 32])
    hp["mlp_reduction"] = 32 # 4 # mlp_reduction

    hp["adapter_start_epoch"] = trial.suggest_int("adapter_start_epoch", 0 if hp["freeze_backbone"] else 1, epochs // 2)
    adapter_start_epoch = hp["adapter_start_epoch"]
    
    batch_size = hp["batch_size"]
    if ds_grad_norm_file: 
        batch_size = 16
    # else:
    #     batch_size = 1 # 16 # 64
    randerase_p = hp["randerase_p"]
    dropout = hp["dropout"]
    weight_decay = hp["weight_decay"]
    mlp_reduction = hp["mlp_reduction"]
    
    # Load small subset of dataset
    train_loader, val_loader, args = get_dataloaders_for(dataset_name, subset_fraction=data_subset, randerase_p=randerase_p,
                                                        batch_size=batch_size, model=model_name)
    # train_loader, val_loader, args = get_subset_dataloaders(subset_fraction=data_subset, dataset_name=dataset_name,
    #                                                        randerase_p=randerase_p)
    # args.num_classes = NUM_CLASSES
    args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    args.batch_size = batch_size
    args.dropout_rate = dropout
    args.lr = lr
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.log_wandb = log_wandb
    args.mlp_reduction = mlp_reduction
    args.adapter_type = hp["adapter_type"]
    args.entropy_coeff = entropy_coeff
    args.adapter_start_epoch = adapter_start_epoch
    args.ds_grad_norm_file = ds_grad_norm_file
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    place_on = hp["place_on"]
    layer_mode = hp["layer_mode"]

    # Notes: backbone_mlp_reduction2 tag for conv/mlp adapters per difference
    # backbone_mlp_reduction3 tag for b5/b6 adapter experts fan out
    if log_wandb:
        wandb.init(
            project="mobilevit-adapter-ablation",
            reinit=True,
            config={
                "lr": round(lr, 4),
                "dropout": round(dropout, 2),
                "randerase_p": round(randerase_p, 2),
                "batch_size": batch_size,
                "data_subset": data_subset,
                "dataset_name": dataset_name,
                "weight_decay": weight_decay,
                "mlp_reduction": mlp_reduction,
                "place_on": place_on,
                "layer_mode": layer_mode,
                "adapter_type": adapter_type,
                "epochs": epochs,
                "B5_adapters": hp["B5_adapters"],
                "B6_adapters": hp["B6_adapters"],
                "adapter_start_epoch": adapter_start_epoch,
                "model_name": model_name,
                "entropy_coeff": entropy_coeff,
                "adapter_start_epoch": adapter_start_epoch,
                "freeze_backbone": freeze_backbone
                
            },
            tags=["cvpr25optuna_birdsmvit", "backbone_mlp_reduction3", f"{dataset_name}_{100*data_subset}p"]
        )
    # Model setup
    # model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=args.num_classes, drop_rate=dropout)
    model = get_model(args)
    unfreeze_last2 = ["stages.3", "stages.4", "head", "adapter"]
    if adapter_cfg is None:
        adapter_cfg = AdapterConfig(
            place_on=place_on, # "mlp",
            layer_mode=layer_mode, # "every",
            every=2
        )
    print('adapter_cfg', adapter_cfg)
    model = inject_adapters_for_model(model, model_name, adapter_cfg, unfreeze_last2, hp, args)
    print("total_params ", sum(p.numel() for p in model.parameters()))
    model.to(args.device)
    
    
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    try:
        val_acc = train_and_eval(model, train_loader, val_loader, args=args, run_test=False, trial=trial if single_objective else None)
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params", total_params)
        last2_adapters, last1_adapters = hp["B5_adapters"], hp["B6_adapters"]
        trial_name = f"B5{last2_adapters}_B6{last1_adapters}_lr{lr:.1e}_do{dropout:.2f}_freeze{freeze_backbone}"
        wandb.config.update({"trial_name": trial_name,
                             "total_params": total_params})
        wandb.log({"val_acc": val_acc})
    except optuna.TrialPruned:
        wandb.log({"status": "pruned"})
        raise
    finally:
        wandb.finish()
    if single_objective:
        return val_acc
    return val_acc, total_params
    
def objective(trial, *, epochs, model_name, unfreeze_layers, adapter_cfg: AdapterConfig,
              seed=42, dataset_name="birds_inat", num_classes=60, subset_fraction=0.25,
              log_wandb=True):
    set_global_seed(seed)
    
    hp = suggest_stage2_hyperparameters(trial, epochs)

    # GPU selection
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trial.number % 2)

    wandb.init(
        project="mobilevit-adapter-ablation",
        reinit=True,
        config={**hp, "model": model_name},
        settings=wandb.Settings(start_method="thread")
    )

    # Load data
    train_loader, val_loader, args = get_subset_dataloaders(
        subset_fraction=subset_fraction,
        dataset_name=dataset_name,
        randerase_p=hp["randerase_p"],
        model=model_name,
        batch_size=hp["batch_size"],
    )
    args.update(hp)
    args.num_classes = num_classes
    args.log_wandb = log_wandb
    device = args.device = "cuda"

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    if "efficientformer" in model_name:
        model = timm.create_model("efficientformerv2_s0", pretrained=True,
                                  num_classes=num_classes, drop_rate=hp["dropout"])

        model = inject_adapters_eformer(
            model=model,
            chain_type=hp["chain_type"],
            config=adapter_cfg,
            experts=(hp["B5_adapters"], hp["B6_adapters"]),
            args=args
        )

    elif "mobilevit" in model_name:
        model = timm.create_model("mobilevitv2_050.cvnets_in1k", pretrained=True,
                                  num_classes=num_classes, drop_rate=hp["dropout"])

        # your mobilevit injector
        from adapters.injector import inject_adapters_mobilevit
        model = inject_adapters_mobilevit(
            model=model,
            chain_type=hp["chain_type"],
            config=adapter_cfg,
            experts=(hp["B5_adapters"], hp["B6_adapters"]),
            args=args
        )

    else:
        raise ValueError(model_name)

    # --------------------------------------------------
    # Freezing / unfreezing
    # --------------------------------------------------
    if hp["freeze_backbone"]:
        for n, p in model.named_parameters():
            p.requires_grad = ("adapter" in n) or n.startswith("head")
    else:
        model = freeze_all_except(model, unfreeze_layers)

    # --------------------------------------------------
    # Optimizer / training
    # --------------------------------------------------
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=hp["lr"]
    )

    val_acc = train_and_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=optimizer,
        device=device,
        trial=trial,
        args=args
    )

    wandb.log({"val_acc": val_acc})
    wandb.finish()

    return val_acc
