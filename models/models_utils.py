import os
import torch
import pickle
import pandas as pd
from shutil import copyfile


def get_pt_path_from_dir(directory, vision_model, report='best val accuracy '):
    """ 
    Given a wandb experiment `directory`, return the saved `vision_model` with best validation accuracy 
    """
    
    cols = ['model', 'accuracy']
    table = []
    # build exp file map
    exp_files = {}
    for d in os.listdir(directory):
        if d.endswith('log'):
            # print(d)
            with open(f'{directory}/{d}', 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('wandb run_id'):
                    wandb_id = line.split(' = ')[-1].strip()
                if line.startswith(report):
                    if wandb_id not in exp_files: exp_files[wandb_id] = [f'{directory}/{d}']
                    val_acc = round(float(line.split(report)[1]), 2)
                    if type(val_acc) == float:
                        # update this to val best
                        exp_files[wandb_id].append(val_acc)
                        # print(wandb_id, exp_files[wandb_id])

    # build results table
    seen_combination = {}
    best_model_path = {}
    # print('exp_files', exp_files)
    for d in os.listdir(directory):
        d = f'{directory}/{d}'
        if os.path.isdir(d):
            # print(d)
            row = []
            exp_args_file = list(filter(lambda x: x.endswith('pickle'), os.listdir(d)))[-1]
            with open(f'{d}/{exp_args_file}', 'rb') as eaf:
                exp_args = pickle.load(eaf)
            wandb_id = exp_args._settings['run_id']
            if wandb_id in exp_files:
                row.append(exp_args.vision_model)
                # row.extend([exp_args.expert_layers, exp_args.mlp_latent_dim, exp_args.moe_router_init])
                row.append(exp_files[wandb_id][-1])
                table.append(row)
                key = row[0]
                # print('row', d)
                # print(wandb_id, key, row, seen_combination)
                if key not in seen_combination: 
                    seen_combination[key] = row[-1]
                    best_model_path[key] = f'{d}/visual_best.pth' # wandb_id
                else:
                    if seen_combination[key] < row[-1]: 
                        seen_combination[key] = row[-1]
                        best_model_path[key] = f'{d}/visual_best.pth' # wandb_id

    # print('final_com', final_com)
    return best_model_path[vision_model]


def save_trial_model(model, trial, epoch, val_loss, val_acc=None, args=None, base_dir=None):
    """
    Save the best model for a trial based on validation loss.
    
    Args:
        model: PyTorch model
        trial: Optuna trial
        epoch: current epoch
        val_loss: validation loss for current epoch
        base_dir: folder to save trial models
    """
    if not trial:
        print('[WARNING]: Not saving model because no trial was found!')
        trial_dir = f"outputs/{args.vision_model}_{args.dataset_name}/"
        os.makedirs(trial_dir, exist_ok=True)
        checkpoint_path = os.path.join(trial_dir, f"checkpoint_best.pth")
        torch.save({'model_state_dict': model.state_dict(), 
                    'args': args, 'val_acc': val_acc, 'val_loss': val_loss, 
                   }, checkpoint_path)
        return
        
    if not base_dir: base_dir = f"outputs/{trial.study.study_name}/"
    # Track best val loss for this trial
    if not hasattr(trial, "_best_val_loss"):
        trial._best_val_loss = float('inf')
        trial._best_model_path = None

    if val_loss < trial._best_val_loss:
        trial._best_val_loss = val_loss
        trial_dir = os.path.join(base_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        checkpoint_path = os.path.join(trial_dir, f"checkpoint_best.pth")
        torch.save({'model_state_dict': model.state_dict(), 
                    'args': args, 'val_acc': val_acc, 'val_loss': val_loss, 
                   }, checkpoint_path)
        trial._best_model_path = checkpoint_path
        trial.set_user_attr("best_model_path", checkpoint_path)
        print(f"[Trial {trial.number}] Saved new best model at epoch {epoch}, val_loss={val_loss:.4f}")

def finalize_study_best_model(study):
    """
    After Optuna study finishes, copy best trial's model to a convenient location.
    
    Args:
        study: Optuna study
        dest_path: path to save the final best model
    """
    dest_path=f"outputs/{study.study_name}/best_model.pth"
    best_trial = study.best_trial
    if "best_model_path" not in best_trial.user_attrs:
        raise ValueError("Best trial does not have a saved model.")
    best_checkpoint = best_trial.user_attrs['best_model_path']
    # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    copyfile(best_checkpoint, dest_path)
    print(f"Copied best trial model from Trial {best_trial.number} to {dest_path}")
    return dest_path