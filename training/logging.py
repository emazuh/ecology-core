import wandb
import pandas as pd
from datetime import datetime
from optuna.importance import get_param_importances
from optuna.trial import TrialState
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def log_train_metrics(epoch, loss, acc, lr):
    wandb.log({
        "train_loss": loss,
        "train_acc": acc,
        "lr": lr
    }, step=epoch)

def log_val_metrics(epoch, loss, acc, entropy=None, prefix="val"):
    log_data = {
        f"{prefix}_loss": loss,
        f"{prefix}_acc": acc
    }
    if entropy is not None:
        log_data[f"{prefix}/entropy"] = entropy

    wandb.log(log_data, step=epoch)

def log_test_metrics(epoch, loss, acc):
    wandb.log({
        "test_loss": loss,
        "test_acc": acc
    }, step=epoch)


def log_optuna_importance_to_wandb(
    study,
    target_idx=0,
    wandb_project="mobilevit-adapter-ablation",
    wandb_entity="ecology-multimodal-2026",
    study_name=None
):
    """
    Logs study-level Optuna hyperparameter importances, best trial, and summary stats to WandB.
    Works for multi-objective studies, and links results to a specific study name.
    """

    # Filter for completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed_trials) < 2:
        print(f"[INFO] Not enough completed trials to compute importances (have {len(completed_trials)}).")
        return None

    # Compute study-level importances
    importances = get_param_importances(
        study,
        target=lambda t: t.values[target_idx]
    )

    df = pd.DataFrame({
        "parameter": list(importances.keys()),
        "importance": list(importances.values())
    })

    # Select best trial for the given objective
    best_trial = max(
        completed_trials,
        key=lambda t: t.values[target_idx] if study.directions[target_idx].name == "maximize" else -t.values[target_idx]
    )

    # Build clear run name
    objective_name = f"obj{target_idx}_{'maximize' if study.directions[target_idx].name == 'maximize' else 'minimize'}"
    study_name = study_name or study.study_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"optuna_summary_{study_name}_{objective_name}_{timestamp}"

    # Initialize WandB
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        reinit=True,
        config={
            "study_name": study_name,
            "objective_index": target_idx,
            "objective_direction": study.directions[target_idx].name,
            "n_completed_trials": len(completed_trials),
            "best_trial_value": best_trial.values[target_idx],
            "best_trial_params": best_trial.params,
        },
        tags=["optuna", "summary", study_name]
    )

    # Log summary info
    wandb.log({
        "param_importance": wandb.Table(dataframe=df),
        "best_trial_value": best_trial.values[target_idx],
    })
    wandb.finish()

    print(f"✅ Logged study-level importances for {study_name} (objective {target_idx})")
    return importances, best_trial

class GateLogger:
    def __init__(self, num_classes=None, log_histogram=False, device='cuda'):
        """
        num_classes: If provided, will compute class-wise affinity.
        log_histogram: Whether to log a histogram of gate activations.
        """
        self.num_classes = num_classes
        self.log_histogram = log_histogram
        self.device = device
        # Initialize affinity tracking
        self.sum_logits = None
        self.counts = None
        self.reset()

    def reset(self):
        self.gate_outputs = []
        if self.num_classes is not None:
            self.sum_logits = None
            self.counts = None

    def hook_fn(self, module, input, output):
        print('calling hookfn')
        if output is None:
            return
        self.gate_outputs.append(output.detach())
        # Only update affinity if labels are available
        if self.num_classes is not None and hasattr(self, "current_labels"):
            self._update_affinity(output, self.current_labels)

    @torch.no_grad()
    def _update_affinity(self, gate_output, labels):
        """Optional: compute class affinity"""
        print('_update_affinity gate_output', len(gate_output))
        if self.sum_logits is None:
            num_experts = gate_output.size(1)
            print('self.num_classes', self.num_classes, 'num_experts', num_experts)
            self.sum_logits = torch.zeros(self.num_classes, num_experts, device=self.device)
            self.counts = torch.zeros(self.num_classes, device=self.device)
        
        for c in range(self.num_classes):
            mask = labels == c
            print("gate_output:", gate_output.shape)
            print("mask sum:", mask.sum().item())
            print("summed:", gate_output[mask].sum(dim=0).shape)
            print("buffer:", self.sum_logits[c].shape)

            if mask.any():
                self.sum_logits[c] += gate_output[mask].sum(dim=(0, 2, 3)) # .sum(dim=0)
                self.counts[c] += mask.sum()

    def get_mean_gates_per_block(self):
        mean_gates_per_block = []
        for b_idx in range(len(self.gate_outputs)):
            block_gates = self.gate_outputs[b_idx]
            if isinstance(block_gates, (list, tuple)):
                # concatenate along batch dimension
                block_gates_cat = torch.cat(block_gates, dim=0)
            else:
                # already a single tensor
                block_gates_cat = block_gates
            mean_gates_per_block.append(block_gates_cat.mean(0))
        return mean_gates_per_block
    
    def compute_entropy(self):
        """Compute batch-wise entropy for each gate output"""
        entropies = []
        for g in self.gate_outputs:
            p = F.softmax(g, dim=1)  # [B, num_experts]
            ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()  # mean entropy over batch
            entropies.append(ent.item())
        return entropies

    def compute_affinity(self, normalize=True):
        if self.num_classes is None or self.sum_logits is None:
            return None
        affinity = self.sum_logits.clone()
        valid = self.counts > 0
        affinity[valid] /= self.counts[valid].unsqueeze(1)
        if normalize:
            row_sums = affinity.sum(dim=1, keepdim=True) + 1e-8
            affinity = affinity / row_sums
        return affinity.cpu().numpy()

    def log_to_wandb(self, step=0, prefix="gates"):
        print('called log_to_wandb')
        # Entropy
        entropies = self.compute_entropy()
        for idx, e in enumerate(entropies):
            wandb.log({f"{prefix}/entropy_block_{idx}": e}, step=step)

        # Class affinity
        if self.num_classes is not None:
            print('num_classes is not None')
            affinity = self.compute_affinity()
            if affinity is not None:
                print('affinity is not None')
                plt.figure(figsize=(8, 6))
                ax = sns.heatmap(affinity.T, annot=False, cmap='viridis')
                plt.title("Class-Expert Affinity")
                wandb.log({f"{prefix}/affinity_map": wandb.Image(plt)}, step=step)
                plt.close()
                print('logged affinity map')

        # Histogram
        if self.log_histogram:
            print('log_histogram is not True')
            all_gates = torch.cat(self.gate_outputs, dim=0).cpu().numpy()
            plt.figure(figsize=(8, 4))
            plt.hist(all_gates.flatten(), bins=64)
            plt.title("Gate Activation Histogram")
            wandb.log({f"{prefix}/histogram": wandb.Image(plt)}, step=step)
            plt.close()
            print('logged gates histogram')