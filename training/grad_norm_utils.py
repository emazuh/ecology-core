# from .grad_norm_utils import aggregate_grad_norms, aggregate_epoch_grad_norms, collect_grad_norms, save_grad_norms_csv
import numpy as np
import pandas as pd

def aggregate_grad_norms(model, eps=1e-12):
    """
    Returns a dictionary of gradient norms aggregated by block/type.
    """
    agg = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        grad_norm = param.grad.norm().item()
        
        # Identify block/type
        if 'head' in name:
            key = 'head'
        elif 'adapter' in name:
            key = 'adapter'
        elif 'transformer' in name:
            if 'attn' in name:
                key = 'transformer_attn'
            elif 'norm' in name:
                key = 'transformer_norm'
            elif 'mlp' in name or 'ffn' in name:
                key = 'transformer_mlp'
            else:
                key = 'transformer_other'
        elif 'conv' in name:
            key = 'conv'
        else:
            key = 'other'
        
        if key not in agg:
            agg[key] = []
        agg[key].append(grad_norm)
    
    # Aggregate by L2 norm over all params in each group
    agg_summary = {k: float(torch.tensor(v).norm().item()) for k, v in agg.items()}
    return agg_summary

def collect_grad_norms(model):
    """
    Collects gradient norms for all parameters in the model for the current batch.
    Returns a dict {param_name: grad_norm}.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

def aggregate_epoch_grad_norms(args, batch_grad_norms):
    """
    Aggregates batch-wise gradient norms into epoch averages.
    batch_grad_norms: list of dicts from collect_grad_norms for each batch
    """
    epoch_avg = {}
    for batch in batch_grad_norms:
        for name, norm in batch.items():
            if name not in epoch_avg:
                epoch_avg[name] = []
            epoch_avg[name].append(norm)
    # Average over batches
    for name in epoch_avg:
        avg_norm = sum(epoch_avg[name]) / len(epoch_avg[name])
        if not hasattr(args, 'grad_norms'):
            args.grad_norms = {}
        if name not in args.grad_norms:
            args.grad_norms[name] = []
        args.grad_norms[name].append(avg_norm)


def save_grad_norms_csv(args, filename="grad_norms_per_layer.csv"):
    """
    Saves the collected grad norms to a CSV file for plotting.
    Handles layers with missing epochs by padding with NaN.
    """
    if hasattr(args, 'grad_norms'):
        # Find max length of any layer
        max_len = max(len(v) for v in args.grad_norms.values())
        # Pad each layer's list to max_len
        padded_dict = {}
        for name, norms in args.grad_norms.items():
            padded = norms + [np.nan] * (max_len - len(norms))
            padded_dict[name] = padded
        grad_df = pd.DataFrame(padded_dict)
        grad_df.index.name = "epoch"
        grad_df.to_csv(filename)
        print(f"[INFO] Grad norms saved to {filename}")
    else:
        print("[WARN] No grad norms found in args.")


