import torch
from ptflops import get_model_complexity_info

def get_model_flops(model, input_res=(3, 224, 224), verbose=False):
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, input_res, as_strings=False, print_per_layer_stat=False, verbose=False
        )
    
        params_m = params / 1e6
        gflops = macs / 1e9  # MACs ~ FLOPs/2, but ptflops reports MACs, so treat as GFLOPs

        if verbose:
            print(f"Parameters: {params_m:.2f} M")
            print(f"Compute: {gflops:.2f} GFLOPs (MACs-based)")
    
    return params_m, gflops