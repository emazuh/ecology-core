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
from adapters.chains import ChainParallelFixed

GATE_MODULES = (
    ChainParallelFixed
)

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

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from collections import defaultdict


class GateLogger:
    def __init__(self, num_classes=None, log_histogram=False, device="cuda"):
        self.num_classes = num_classes
        self.log_histogram = log_histogram
        self.device = device
        self.reset()

    def reset(self):
        # block_idx -> list of tensors ([E] or [B,E])
        self.gate_outputs = defaultdict(list)

        # block_idx -> [num_classes, E]
        self.sum_logits = {} if self.num_classes is not None else None
        self.counts = {} if self.num_classes is not None else None

    # ------------------------------------------------
    # Recording API (called from forward())
    # ------------------------------------------------
    @torch.no_grad()
    def record(self, block_idx, gates, labels=None):
        """
        gates:
            fixed: [E]
            input-dependent: [B, E]
        labels: [B] or None
        """
        gates = gates.detach().to(self.device)

        # normalize shape → [B, E]
        if gates.dim() == 1:
            gates = gates.unsqueeze(0)

        self.gate_outputs[block_idx].append(gates.cpu())

        if self.num_classes is not None and labels is not None:
            self._update_affinity(block_idx, gates, labels)

    # ------------------------------------------------
    # Affinity
    # ------------------------------------------------
    @torch.no_grad()
    def _update_affinity(self, block_idx, gates, labels):
        # gates: [B, E]
        B, E = gates.shape

        if B == 1:
            # no affinity for fixed gates
            return

        if block_idx not in self.sum_logits:
            self.sum_logits[block_idx] = torch.zeros(
                self.num_classes, E, device=self.device
            )
            self.counts[block_idx] = torch.zeros(
                self.num_classes, device=self.device
            )

        # print("unique labels seen:", labels.unique())
        
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                self.sum_logits[block_idx][c] += gates[mask].sum(dim=0)
                self.counts[block_idx][c] += mask.sum()

    def compute_affinity(self, normalize=True):
        if self.num_classes is None:
            return None

        affinities = {}
        for b in self.sum_logits:
            A = self.sum_logits[b].clone()
            counts = self.counts[b]

            valid = counts > 0
            A[valid] /= counts[valid].unsqueeze(1)

            if normalize:
                A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

            affinities[b] = A.cpu().numpy()
        return affinities

    # ------------------------------------------------
    # Logging
    # ------------------------------------------------
    def log_to_wandb(self, step=0, prefix="gates"):
        # ---------- Histogram (per block) ----------
        if self.log_histogram:
            for b, gate_list in self.gate_outputs.items():
                values = torch.cat([g.flatten() for g in gate_list])
                plt.figure(figsize=(6, 4))
                plt.hist(values.numpy(), bins=64)
                plt.title(f"Gate Histogram – Block {b}")
                wandb.log(
                    {f"{prefix}/hist_block_{b}": wandb.Image(plt)},
                    step=step
                )
                plt.close()

        # ---------- Affinity ----------
        if self.num_classes is not None:
            affinities = self.compute_affinity()
            for b, A in affinities.items():
                plt.figure(figsize=(6, 5))
                sns.heatmap(A.T, cmap="viridis")
                plt.title(f"Class–Expert Affinity – Block {b}")
                wandb.log(
                    {f"{prefix}/affinity_block_{b}": wandb.Image(plt)},
                    step=step
                )
                plt.close()
