import sys
sys.path.insert(0, '..')
from adapters.chains import ChainSequential, ChainParallelFixed, ChainParallelInputDependent, ChainSequentialInputDependent
from hpo.objective import set_seed

import wandb
import torch
import torch.nn as nn
from dataloaders.dataloaders import get_dataloaders_for
from models import get_model
from adapters.injector import inject_adapters_for_model
from configs.adapter_config import AdapterConfig
from training.engine import train_and_eval


MULTI_DS_SUBSET = 1.0 # 1.0 #

best_hyperparams = {'lr': 0.0008024521889759168, 'batch_size': 256, 'dropout': 0.24823493843102154, 
                    'randerase_p': 0.35941761914801756, 'warmup_ratio': 0.07266791504808244, "weight_decay": 0.0}


# --- Define model variants ---

# Adapter hyperparameters
best_adapter_hyperparams = {'B5_adapters': 1, 'B6_adapters': 2, 'freeze_backbone': False, 'input_dependent_gating': True,
                            'entropy_coeff': 0.00045086724350302073, 'max_grad_norm': 1.9760255312955526, 
                            'adapter_type': 'simple', 'adapter_start_epoch': 17, 'chain_type': 'par_fixed', # 'seq'
                           }

best_adapter_hyperparams['adapter_scale'] = 1.0 # {'adapter_scale': 1.0}
B5_adapters = getattr(best_adapter_hyperparams, 'B5_adapters', 0)
B6_adapters = best_adapter_hyperparams['B6_adapters']
freeze_backbone = best_adapter_hyperparams['freeze_backbone']
input_dependent_gating = best_adapter_hyperparams['input_dependent_gating']
adapter_type = best_adapter_hyperparams['adapter_type']
# Only suggest reduction if bottleneck is chosen
reduction = None
if adapter_type == "bottleneck":
    reduction = best_adapter_hyperparams['reduction']

place_on = "mlp" # "conv" # best_adapter_hyperparams['place_on'] # place_on, # "mlp",
layer_mode = "every" # best_adapter_hyperparams['layer_mode'] # layer_mode, # "every",

MODEL_VARIANTS = {
    "eformer_adapters_2x1": {
        "model_name": "efficientformer-s0",
        "adapter_cfg": {"experts_last2": (B5_adapters, B6_adapters), "reduction": reduction,
                                  "input_dependent_gating": input_dependent_gating},
        "description": "Efficientformer + adapters baseline"
    },
    "mobilevit_adapters_2x1": { # mobilevit_adapters_4x4
        "model_name": "mobilevit",
        "adapter_cfg": {"experts_last2": (B5_adapters, B6_adapters), "reduction": reduction,
                                  "input_dependent_gating": input_dependent_gating},
        "description": "MobileViT + adapters (2×2)"
    },
    "efficientformer-s0": {
        "model_name": "efficientformer-s0",
        "adapter_cfg": None,
        "description": "Efficientformer s0"
    },
    "mobilevit-0.5": {
        "model_name": "mobilevit-0.5",
        "adapter_cfg": None,
        "description": "MobileViTV2 0.5"
    },
    # "efficientformer-s1": {
    #     "model_name": "efficientformer-s1",
    #     "adapter_cfg": None,
    #     "description": "Efficientformer s1"
    # },
    # "mobilevit-1.0": {
    #     "model_name": "mobilevit-1.0",
    #     "adapter_cfg": None,
    #     "description": "MobileViTV2 1.0"
    # },
}

import os
def run_main_table(seed=42, epochs=20):
    set_seed(seed)
    
    lr = best_hyperparams["lr"]
    batch_size = 256 # 128 # best_hyperparams["batch_size"]
    dropout = best_hyperparams["dropout"]
    randerase_p = best_hyperparams["randerase_p"]
    weight_decay = best_hyperparams["weight_decay"]

    chain_type = best_adapter_hyperparams['chain_type']
    chain_map = {
        "seq": ChainSequential,
        "seq_input": ChainSequentialInputDependent,
        "par_fixed": ChainParallelFixed,
        "par_input": ChainParallelInputDependent
    }
    chain_cls = chain_map[chain_type]
    
    adapter_start_epoch = best_adapter_hyperparams['adapter_start_epoch']
    
    # train_loader, val_loader, args = get_subset_dataloaders(subset_fraction=MULTI_DS_SUBSET, dataset_name=DATASET_NAME)
    
    # --- Define datasets to evaluate ---
    datasets = ["birds_inat" , "rare_species", "iwildcam", "ssw60"]

    warmup_ratio = best_hyperparams["warmup_ratio"]
    entropy_coeff = best_adapter_hyperparams['entropy_coeff']
    max_grad_norm = best_adapter_hyperparams['max_grad_norm']
    adapter_scale = best_adapter_hyperparams['adapter_scale']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    # --- Iterate over datasets and model variants ---
    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        train_loader, val_loader, args = get_dataloaders_for(dataset_name, subset_fraction=MULTI_DS_SUBSET)
        args.entropy_coeff = entropy_coeff
        
        args.max_grad_norm = max_grad_norm
        args.adapter_scale = adapter_scale
        args.dataset_name = dataset_name
        # args.num_classes = NUM_CLASSES
        args.lr = lr
        args.epochs = epochs
        args.weight_decay = weight_decay
        args.warmup_ratio = warmup_ratio
        args.dropout_rate = dropout
        args.adapter_type = adapter_type
    
        args.device = device = "cuda" if torch.cuda.is_available() else "cpu"
    
        dataset_results = {}

        # --- WandB setup ---
    
        for model_key, cfg in MODEL_VARIANTS.items():
            print(f"\n🚀 Training {model_key} on {dataset_name}")
    
            args.model = args.vision_model = cfg["model_name"]
            if "adapter" in cfg["model_name"]:
                args.adapter_start_epoch = adapter_start_epoch

            mdl_cfg = {
                    "dataset_name": dataset_name,
                    "subset": MULTI_DS_SUBSET,
                    "lr": round(lr, 5),
                    "dropout": round(dropout, 5),
                    "randerase_p": round(randerase_p, 5),
                    "batch_size": batch_size,
                    "warmup_ratio": warmup_ratio,
                    "model_key": model_key,
                    "model_name": args.model,
                    "epochs": epochs,
                    # TODO: Also log the adapter model configs
                }

            run = wandb.init(
                project="mobilevit-adapter-ablation",
                reinit=True,
                config=mdl_cfg,
                tags=["main-table-dev", "baselines", "all_vision_variants", "rare_wild_birds_adpt", "with_fourier", # small_variants
                      "mobilevit-multidataset-eval", f"{dataset_name}_{100*MULTI_DS_SUBSET}p"]
            )
            
            model = get_model(args)
    
            # Inject adapters if needed
            if cfg["adapter_cfg"] is not None:
                unfreeze_last2 = ["stages.3", "stages.4", "head", "adapter"]
                if 'mobilevit' in cfg["model_name"]:
                    unfreeze_last2 = ["stages.3", "stages.4", "head", "adapter"] # stem ?
                else: 
                    unfreeze_last2 = ["stages.2.blocks", "stages.3.blocks", "head", "adapter"] # "stem", ?

                adapter_cfg = AdapterConfig(
                    place_on=place_on,
                    layer_mode=layer_mode,
                    every=2
                )
                model = inject_adapters_for_model(model, cfg["model_name"], adapter_cfg, unfreeze_last2, best_adapter_hyperparams, args)
                
                # # model = inject_adapters_after_mlp(model, **cfg["adapter_cfg"])
                # if "mobilevit" in args.model.lower():
                #     # model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=args.num_classes, drop_rate=dropout)
                #     model = inject_adapters_after_ffn_mvit(model, chain_cls=chain_cls,
                #                                            experts_last2=(B5_adapters, B6_adapters), args=args)
                # elif "efficientformer" in args.model.lower():
                #     # model = timm.create_model('efficientformerv2_s0', pretrained=True, num_classes=args.num_classes, drop_rate=dropout)
                #     # print('chain_cls', chain_cls)
                #     model = inject_adapters_after_ffn_eformer(model, chain_cls=chain_cls, 
                #                                               experts_last2=(B5_adapters, B6_adapters), args=args)
                # else:
                #     raise ValueError(f"Unsupported model: {model_name}")
        
                # if freeze_backbone:
                #     for name, param in model.named_parameters():
                #         param.requires_grad = ("adapter" in name) or name.startwith("head")
                # else:
                #     # if 'mobilevit' in cfg["model_name"]:
                #     #     unfreeze_last2 = ["stages.3", "stages.4", "head", "adapter"] 
                #     # else: 
                #     #     unfreeze_last2 = ["stages.2.blocks", "stages.3.blocks", "head", "adapter"]
                #     model = freeze_all_except(model, unfreeze_last2)
            
                if not freeze_backbone and adapter_start_epoch > 0:
                    for name, param in model.named_parameters():
                        # freeze adapters if explicitly requested to start training after adapter_start_epoch
                        if ("adapter" in name): param.requires_grad = False   
                
    
            model.to(device)
            # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            # args.log_wandb = True
            val_acc = train_and_eval(model, train_loader, val_loader, args=args, run_test=True)
    
            dataset_results[model_key] = val_acc
            wandb.log({f"{dataset_name}/{model_key}_val_acc": val_acc})
            
            model_folder = f"./ckpts/{dataset_name}"
            model_path = f"{model_folder}/{model_key}.pth"
            os.makedirs(model_folder, exist_ok=True)
            torch.save(model.state_dict(), model_path) #  PyTorch

            # Create an artifact and add the model file
            artifact = wandb.Artifact(name=f"{dataset_name}.{model_key}", type="model")
            artifact.add_file(model_path)
            # wandb.save(model_path)
            print('[INFO]: Saved model to', model_path, 'and wandb artifact', f'{dataset_name}.{model_key}')

            # Log the artifact to W&B
            run.log_artifact(artifact)
    
        results[dataset_name] = dataset_results
    
    wandb.finish()
    
    # --- Summary ---
    print("\n=== Multi-Dataset Summary ===")
    for dataset_name, dataset_results in results.items():
        print(f"\n📊 Dataset: {dataset_name}")
        baseline_acc = dataset_results["mobilevit-0.5"] # mobilevit_baseline
        for model_key, acc in dataset_results.items():
            delta = acc - baseline_acc
            print(f"  {model_key:30s} → Acc: {acc:.3f}  | Δ vs baseline: {delta:+.3f}")
    return results

# if __name__ == '__main__':
import pickle
results = run_main_table(epochs=20)
with open(f'save_results_rare_wild_birds_adpt_ds{100*MULTI_DS_SUBSET}p_v2.pkl', 'wb') as f:
    pickle.dump(results, f)
