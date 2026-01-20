import wandb
import optuna
import math
import torch
from tqdm import tqdm

from .utils import flatten_video_batch, unflatten_video_logits, build_subset_loader
from .scheduler_utils import build_optimizer, build_scheduler, update_scheduler_per_step
from .losses import EntropyCELoss
from .logging import log_train_metrics, log_val_metrics, log_test_metrics, GateLogger
from .grad_norm_utils import aggregate_grad_norms, aggregate_epoch_grad_norms, collect_grad_norms, save_grad_norms_csv

import os
script_directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, os.path.join(script_directory, '..'))
from models.models_utils import save_trial_model
from adapters.chains import ChainParallelFixed

def run_one_epoch(model, loader, optimizer, scheduler, args, epoch, train=True, gate_logger=None):
    model.train() if train else model.eval()

    total_loss, correct, total = 0, 0, 0
    entropy_values = []
    # iter_per_epoch = len(loader)
    iter_per_epoch = math.ceil(len(loader.dataset)/args.batch_size)

    criterion = torch.nn.CrossEntropyLoss() # EntropyCELoss()

    batch_grad_norms = []
    if train and args.ds_grad_norm_file: batch_grad_norms.append(collect_grad_norms(model))
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(args.device), y.to(args.device)

        if train:
            optimizer.zero_grad()

        if gate_logger:
            for name, module in model.named_modules():
                # print(name, type(module))
                if isinstance(module, ChainParallelFixed):
                    module.register_forward_hook(gate_logger.hook_fn)
            
            # for logging expert gate information
            gate_logger.reset()
            args.gates_ent_loss = 0 # reset before forward
            gate_logger.current_labels = y

        # ----- video flatten -----
        is_video = (X.ndim == 5)
        if is_video:
            X, shape = flatten_video_batch(X)

        logits = model(X)

        if is_video:
            logits = unflatten_video_logits(logits, shape)

        # loss
        # loss = criterion(logits, y, args, include_entropy=train or args.apply_entropy_to_val)

        loss = criterion(logits, y)

        # Entropy regularization
        # print('gate_outputs', len(gate_logger.gate_outputs), epoch, args.adapter_start_epoch)
        if hasattr(args, 'adapter_start_epoch') and epoch >= args.adapter_start_epoch:
            if hasattr(args, 'gates_ent_loss'): 
                entropy_value = args.gates_ent_loss
                loss += entropy_value
                # print('entropy_value', -entropy_value, getattr(args, 'entropy_coeff', 1e-4))
                # Log entropy per block to W&B
                if args.log_wandb and train:
                    # plot negative of the entropy value since we negated it to add to the loss 
                    # for entropy maximization (plot the actual entropy computed)
                    # print('entropy_value', -entropy_value)
                    entropy_value = entropy_value if isinstance(entropy_value, int) else entropy_value.item()
                    wandb.log({f"entropy/train_ent": -entropy_value}, commit=False)
        if train:
            loss.backward()

            for name, param in model.named_parameters():
                if "adapter" in name or "gates" in name:
                    # print(name, param.shape, param.requires_grad, args.vision_model)
                    if param.grad is not None:
                        param.grad.data.mul_(getattr(args, 'adapter_scale', 15.0))
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm
            )

            optimizer.step()
            # update_scheduler_per_step(scheduler, epoch, iter_per_epoch, batch_idx, args)
            scheduler.step(epoch) # args.steps) # epoch
            if args.log_wandb:
                # print('dir(scheduler)', dir(scheduler))
                if hasattr(scheduler, 'get_last_lr') and log_wandb:
                    wandb.log({'lr': scheduler.get_lr()[0]}, commit=False)
                    # print('scheduler.get_lr()[0]', scheduler.get_last_lr()[0])
                elif hasattr(scheduler, 'get_lr') and log_wandb:
                    wandb.log({'lr': scheduler.get_lr()[0]}, commit=False)
                    # print('scheduler.get_lr()[0]', scheduler.get_lr()[0])
                else:
                    wandb.log({'lr': optimizer.param_groups[0]["lr"]}, commit=False)
                    # print('optimizer.param_groups[0]["lr"]', optimizer.param_groups[0]["lr"])
        else:
            # Update affinity logger during validation
            if gate_logger: #  and gate_logger.gate_outputs:
                # outputs here should be per-expert logits
                for block_gates in gate_logger.gate_outputs:
                    gate_logger._update_affinity(block_gates, y)
        
        
        # stats
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

        # entropy_values.append(args.gates_ent_loss)

    avg_loss = total_loss / total
    avg_acc = correct / total
    avg_entropy = [] # torch.tensor(entropy_values).float().mean().item() # .detach()

    if train and args.ds_grad_norm_file: aggregate_epoch_grad_norms(args, batch_grad_norms)

    if args.log_wandb:
        # Compute and log affinity once per epoch
        if gate_logger: print('gate_outputs', len(gate_logger.gate_outputs))
        if gate_logger and gate_logger.gate_outputs:
            print('calling log_to_wandb')
            gate_logger.log_to_wandb(step=epoch, prefix="val_routing")
    return avg_loss, avg_acc, avg_entropy


def train_and_eval(model, train_loader, val_loader, args, run_test=False, trial=None):

    model.to(args.device)

    # --------------------------
    # Build optimizer & scheduler
    # --------------------------
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    best_val = 0
    
    gate_logger = GateLogger(num_classes=args.num_classes, log_histogram=True)

    # Optional subset loader
    subset_loader = build_subset_loader(val_loader, samples_per_class=args.val_samples_per_class)

    epochs = args.epochs
    for epoch in range(args.epochs):

        # unfreeze adapters
        if hasattr(args, "adapter_start_epoch") and epoch == args.adapter_start_epoch:
            print(f"[INFO] Unfreezing adapters at epoch {epoch}")
            for n, p in model.named_parameters():
                if "adapter" in n:
                    p.requires_grad = True

            optimizer = build_optimizer(model, args.lr, args.weight_decay)
            scheduler = build_scheduler(args, optimizer)  # restart LR schedule

        # ---------------------------
        # Train
        # ---------------------------
        train_loss, train_acc, train_ent = run_one_epoch(
            model, train_loader, optimizer, scheduler, args, epoch, train=True
        )

        # log train
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        if args.log_wandb:
            lr = optimizer.param_groups[0]["lr"]
            log_train_metrics(epoch, train_loss, train_acc, lr)
            #     if hasattr(scheduler, 'get_last_lr') and log_wandb:
            #         # wandb.log({'lr': scheduler.get_last_lr()[0]}, commit=False)
            #         lr = scheduler.get_last_lr()[0]
            #     elif hasattr(scheduler, 'get_lr') and log_wandb:
            #         # wandb.log({'lr': scheduler.get_lr()[0]}, commit=False)
            #         lr = scheduler.get_lr()[0]

        # ---------------------------
        # Subset validation
        # ---------------------------
        val_loss, val_acc, val_ent = run_one_epoch(
            model,
            subset_loader,
            optimizer=None,
            scheduler=None,
            args=args,
            epoch=epoch,
            train=False,
            gate_logger=gate_logger
        )

        print(f"Validation (subset) | Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")
        if args.log_wandb:
            log_val_metrics(epoch, val_loss, val_acc, val_ent, prefix="val_subset")

        if val_acc > best_val:
            best_val = val_acc

        # save_trial_model(model, trial, epoch, val_loss, val_acc=val_acc, args=args)

        # --- Report intermediate value to Optuna ---
        if trial is not None:
            trial.report(val_acc, step=epoch)
            
            # Optionally prune
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if args.ds_grad_norm_file: save_grad_norms_csv(args, args.ds_grad_norm_file)
    
    # ---------------------------
    # Full validation after training
    # ---------------------------
    full_val_loss, full_val_acc, full_val_ent = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        scheduler=None,
        args=args,
        epoch=args.epochs - 1,
        train=False
    )

    print(f"✅ Full Validation | Loss: {full_val_loss:.4f} | Acc: {full_val_acc:.3f}")
    if args.log_wandb:
        log_val_metrics(args.epochs - 1, full_val_loss, full_val_acc, full_val_ent, prefix="val_full")

    # if testing log_optuna_importance_to_wandb
    if run_test:
        full_test_loss, full_test_acc, full_test_ent = run_one_epoch(
            model,
            args.test_loader,
            optimizer=None,
            scheduler=None,
            args=args,
            epoch=args.epochs - 1,
            train=False
        )
        print(f"✅ Full Test | Loss: {full_test_loss:.4f} | Acc: {full_test_acc:.3f}")
        if args.log_wandb:
            log_val_metrics(args.epochs - 1, full_test_loss, full_test_acc, full_test_ent, prefix="test")
    
    return full_val_acc
