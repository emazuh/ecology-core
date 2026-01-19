import os
import json
from .objective import *

script_directory = os.path.dirname(os.path.realpath(__file__))

def objective_adapter_layers(trial, adapter_cfg, seed=42, epochs=20, data_subset=0.1, dataset_name="birds_inat", model_name="mobilevit",
                  log_wandb=True, chain_type="par_fixed", ds_grad_norm_file=None, single_objective=False, adapter_type=None):
    
    set_seed(seed)
    
    with open(os.path.join(script_directory, '../configs/general_config.json'), 'r') as f:
        hp = json.load(f)

    lr = hp["lr"]
    hp["place_on"], hp["layer_mode"] = None, None
    if adapter_cfg is None:
        hp["place_on"] = trial.suggest_categorical("place_on", ["mlp", "conv"])
        hp["layer_mode"] = trial.suggest_categorical("layer_mode", ["every", "all"])

    if adapter_type is None:
        adapter_type = hp["adapter_type"] = trial.suggest_categorical("adapter_type", ["simple", "fourier", "bottleneck"]) #  
    else:
        hp["adapter_type"] = adapter_type

    reduction = None
    if adapter_type == "bottleneck":
        reduction = trial.suggest_categorical("reduction", [4, 8, 16, 32])
    
    entropy_coeff = hp["entropy_coeff"] = trial.suggest_categorical("entropy_coeff", [1e-4,  1e-2, 1e-3]) # # -1e-2, , -1e-3 -0.1, 
    
    # TODO: Search over these in stage 2
    hp["chain_type"] = chain_type
    
    # Adapter fan-out
    hp["B5_adapters"] = trial.suggest_categorical("B5_adapters", [1, 2]) # 4, 8
    hp["B6_adapters"] = trial.suggest_categorical("B6_adapters", [1, 2, 4]) # , 8
    freeze_backbone = hp["freeze_backbone"] = trial.suggest_categorical("freeze_backbone", [False, True]) # False # True
    
    # mlp_reduction = trial.suggest_categorical("mlp_reduction", [2, 4, 8, 16, 32])
    hp["mlp_reduction"] = 1 # 32 # 4 # mlp_reduction # Go easy for now 

    hp["adapter_start_epoch"] = trial.suggest_int("adapter_start_epoch", 0 if hp["freeze_backbone"] else epochs // 3, epochs // 2) # else 1
    adapter_start_epoch = hp["adapter_start_epoch"]
    adapter_scale = hp["adapter_scale"] = trial.suggest_categorical("adapter_scale", [1.0, 2.0, 10.0]) # , 0.1
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.1, 0.01, 1.0, 0.3, 5.0])
    
    batch_size = hp["batch_size"]

    randerase_p = hp["randerase_p"]
    dropout = hp["dropout"]
    weight_decay = hp["weight_decay"]
    mlp_reduction = hp["mlp_reduction"]
    
    # Load small subset of dataset
    train_loader, val_loader, args = get_dataloaders_for(dataset_name, subset_fraction=data_subset, randerase_p=randerase_p,
                                                        batch_size=batch_size, model=model_name)

    args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1) # deprecated in favor of timm scheduler warmup epochs
    args.batch_size = batch_size
    args.dropout_rate = dropout
    args.lr = lr
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.log_wandb = log_wandb
    args.mlp_reduction = mlp_reduction
    args.adapter_scale = adapter_scale
    args.adapter_type = hp["adapter_type"]
    args.reduction = reduction
    args.entropy_coeff = entropy_coeff
    args.adapter_start_epoch = adapter_start_epoch
    args.ds_grad_norm_file = ds_grad_norm_file
    args.max_grad_norm = max_grad_norm
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.hp = hp
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
                "reduction": reduction if reduction else "N/A",
                "epochs": epochs,
                "B5_adapters": hp["B5_adapters"],
                "B6_adapters": hp["B6_adapters"],
                "adapter_start_epoch": adapter_start_epoch,
                "model_name": model_name,
                "entropy_coeff": entropy_coeff,
                "adapter_start_epoch": adapter_start_epoch,
                "freeze_backbone": freeze_backbone,
                "adapter_scale": adapter_scale,
                "chain_type": chain_type,
                "max_grad_norm": max_grad_norm
                
            },
            tags=["cvpr25optuna_birdsmvit", f"{dataset_name}_{100*data_subset}p", # "backbone_mlp_reduction3", 
                  "layers_objective", "hpo_2"
                 ]
            # Change log
            # hpo_0: original
            # hpo_1: removed adapter_scale = 0.1 causing Nan
            # hpo_2: removed entropy_coeff = 0.1 unstable training
        )
    # Model setup
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
    
    
    try:
        val_acc = train_and_eval(model, train_loader, val_loader, args=args, run_test=False, trial=trial if single_objective else None)
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params", total_params)
        last2_adapters, last1_adapters = hp["B5_adapters"], hp["B6_adapters"]
        trial_name = f"B5{last2_adapters}_B6{last1_adapters}_lr{lr:.1e}_do{dropout:.2f}_freeze{freeze_backbone}"
        if log_wandb:
            wandb.config.update({"trial_name": trial_name,
                                 "total_params": total_params})
            wandb.log({"val_acc": val_acc})
    except optuna.TrialPruned:
        if log_wandb: wandb.log({"status": "pruned"})
        raise
    finally:
        if log_wandb: wandb.finish()
    if single_objective:
        return val_acc
    return val_acc, total_params




if __name__ == '__main__':
    # mvit
    study_name = "cvpr_general_hyperparams_birdsmvit"
    study_mlp_layers = optuna.create_study(#directions=["maximize", "minimize"], study_name=study_name,
                                           direction="maximize", study_name=study_name,
                                           pruner=optuna.pruners.PercentilePruner(
                                                    percentile=15.0,     # prune bottom 25% of trials
                                                    n_startup_trials=2,  # allow some trials to complete fully
                                                    n_warmup_steps=5,    # wait 1 epoch before starting pruning
                                                ))
    study_mlp_layers.optimize(
            lambda trial: objective_adapter_layers(
                trial,
                adapter_cfg=None,
                epochs=3, # 20
                model_name="mobilevit",
                dataset_name="birds_inat",
                data_subset=0.05, #0.25,#,0.25,
                log_wandb=True,#True,
                chain_type="seq_input",#"par_fixed",
                ds_grad_norm_file=None,# 'birds_mvit_grad_norm_per_layer.csv',
                single_objective=True,
            ),
            n_trials=3, #100
        )


    # eformer 
    study_name = "cvpr_general_hyperparams_birdseformer"
    study_mlp_layers = optuna.create_study(#directions=["maximize", "minimize"], study_name=study_name,
                                           direction="maximize", study_name=study_name,
                                           pruner=optuna.pruners.PercentilePruner(
                                                    percentile=15.0,     # prune bottom 25% of trials
                                                    n_startup_trials=2,  # allow some trials to complete fully
                                                    n_warmup_steps=5,    # wait 1 epoch before starting pruning
                                                ))
    study_mlp_layers.optimize(
            lambda trial: objective_adapter_layers(
                trial,
                adapter_cfg=None,
                epochs=3, # 20
                model_name="efficientformer",
                dataset_name="birds_inat",
                data_subset=0.05, #0.25,#,0.25,
                log_wandb=False,#True,
                chain_type="seq_input",#"par_fixed",
                single_objective=True,
            ),
            n_trials=3, #100
        )