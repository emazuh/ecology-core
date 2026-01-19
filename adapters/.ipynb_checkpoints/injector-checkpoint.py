import torch
import torch.nn as nn
from .registry import CHAIN_MAP
from configs.adapter_config import AdapterConfig
from .freeze import freeze_all_except

# ------------------------------------------------------------------
# Generic layer selector all, every, last_k
# ------------------------------------------------------------------
def select_layers(num_layers, mode, layers=None, every=2, last_k=None):
    if mode == "all":
        return list(range(num_layers))
    if mode == "every":
        return list(range(0, num_layers, every))
    if mode == "last_k":
        return list(range(num_layers - last_k, num_layers))
    if mode == "custom":
        return layers
    raise ValueError(f"Invalid mode {mode}")


def compress_eformer_mlp(comp_block, args):
    # always reduce MLP and freeze [finding which /C works best, lowest without acc drop] [entire last 2 blocks]
    # if this doesn't catch always, add another for loop over modules
    # if mlp_reduction: 2,8,16
    # if do_low_rank_approx:
    #     # compute low rank approximation of conv weights
    #     m.mlp.fc1.conv = compress_layer(m.mlp.fc1.conv, low_rank_dim=args.mlp_latent_dim, device=args.device)
    #     m.mlp.fc2.conv = compress_layer(m.mlp.fc2.conv, low_rank_dim=args.mlp_latent_dim, device=args.device)
    # for n, m in comp_model.named_modules():
    #     if isinstance(m, EfficientFormerV2Block) and 
    if hasattr(args, "mlp_reduction") and args.mlp_reduction >= 1:
        out_channels = comp_block.mlp.fc2.conv.out_channels
        hidden_size = max(2, int(out_channels // args.mlp_reduction))
        print('compressing mlp size to', hidden_size, 'from', out_channels, 'args.mlp_reduction', args.mlp_reduction)
        comp_block.mlp.fc1.conv = compress_layer(comp_block.mlp.fc1.conv, low_rank_dim=hidden_size, device=args.device)
        comp_block.mlp.fc2.conv = compress_layer(comp_block.mlp.fc2.conv, low_rank_dim=hidden_size, device=args.device)
            
# ------------------------------------------------------------------
# Generic transformer injector (EfficientFormer)
# ------------------------------------------------------------------
# def inject_adapters_eformer(model, chain_type, config: AdapterConfig, experts=(1, 1), args=None):
    
#     chain_cls = CHAIN_MAP[chain_type]

#     stages = ["stages.2.blocks", "stages.3.blocks"]
#     for si, name in enumerate(stages):
#         blocks = dict(model.named_modules())[name]
#         n = len(blocks)
#         inject_idxs = select_layers(
#             num_layers=n,
#             mode=config.layer_mode,
#             layers=config.layer_indices,
#             every=config.every,
#             last_k=config.last_k
#         )
#         experts_per_block = experts[si]

#         for idx in inject_idxs:
#             block = blocks[idx]

#             # print('idx, block', idx, type(block))
            
#             # MLP
#             if config.place_on in ("mlp", "both"):
#                 # compress mlp size if needed
                
                
#                 conv = block.mlp.fc2.conv
#                 adapters = chain_cls(
#                     in_channels=conv.out_channels,
#                     num_blocks=1,
#                     experts_per_block=[experts_per_block],
#                     args=args
#                 )
#                 compress_eformer_mlp(block, args)
                
#                 # print('type(block.mlp.fc2.conv)', type(block.mlp.fc2.conv))
#                 # for mname, mm in block.mlp.fc2.conv.named_modules():
#                 #     print('mname', mname)
#                 #     break
#                 block.mlp.fc2.conv = nn.Sequential(conv, adapters)

#             # Conv mixer
#             if config.place_on in ("conv", "both") and hasattr(block, "token_mixer"):
#                 if hasattr(block.token_mixer, "talking_head2"):
#                     conv = block.token_mixer.talking_head2
#                     adapters = chain_cls(
#                         in_channels=conv.out_channels,
#                         num_blocks=1,
#                         experts_per_block=[experts_per_block],
#                         args=args
#                     )
#                     # print('type(block.token_mixer.talking_head2)', type(block.token_mixer.talking_head2))
#                     # for mname, mm in block.token_mixer.talking_head2.named_modules():
#                     #     print('mname', mname)
#                     #     break
#                     block.token_mixer.talking_head2 = nn.Sequential(conv, adapters)

#     return model

def inject_adapters_eformer(model, chain_type, config: AdapterConfig, experts=(1, 1), args=None):

    chain_cls = CHAIN_MAP[chain_type]

    target_stages = ["stages.2.blocks", "stages.3.blocks"]

    for si, name in enumerate(target_stages):

        blocks = dict(model.named_modules())[name]
        n = len(blocks)

        inject_idxs = select_layers(
            num_layers=n,
            mode=config.layer_mode,
            layers=config.layer_indices,
            every=config.every,
            last_k=config.last_k
        )
        experts_per_block = experts[si]

        for idx in inject_idxs:
            block = blocks[idx]

            # ---------------- MLP ----------------
            if config.place_on in ("mlp", "both"):
                conv = block.mlp.fc2.conv

                adapters = chain_cls(
                    in_channels=conv.out_channels,
                    num_blocks=1,
                    experts_per_block=[experts_per_block],
                    args=args
                )

                compress_eformer_mlp(block, args)

                class WrappedFC2(nn.Module):
                    def __init__(self, conv, adapter):
                        super().__init__()
                        self.conv = conv
                        self.adapter = adapter

                    def forward(self, x):
                        y = self.conv(x)
                        return y + self.adapter(y)

                block.mlp.fc2.conv = WrappedFC2(conv, adapters)

            # ------------- Conv mixer (talking-head2) -------------
            if config.place_on in ("conv", "both") and hasattr(block, "token_mixer"):
                if hasattr(block.token_mixer, "talking_head2"):

                    conv = block.token_mixer.talking_head2

                    adapters = chain_cls(
                        in_channels=conv.out_channels,
                        num_blocks=1,
                        experts_per_block=[experts_per_block],
                        args=args
                    )

                    class WrappedMixer(nn.Module):
                        def __init__(self, conv, adapter):
                            super().__init__()
                            self.conv = conv
                            self.adapter = adapter

                        def forward(self, x):
                            y = self.conv(x)
                            return y + self.adapter(y)

                    block.token_mixer.talking_head2 = WrappedMixer(conv, adapters)

    return model

def compress_mvit_mlp(comp_block, args):

    if args.mlp_reduction >= 1:
        out_channels = comp_block.mlp.fc2.out_channels
        hidden_size = max(2, int(out_channels // args.mlp_reduction))
        print('compressing mlp size to', hidden_size, 'from', out_channels, 'args.mlp_reduction', args.mlp_reduction)
        comp_block.mlp.fc1 = compress_layer(comp_block.mlp.fc1, low_rank_dim=hidden_size, device=args.device)
        comp_block.mlp.fc2 = compress_layer(comp_block.mlp.fc2, low_rank_dim=hidden_size, device=args.device)

# def inject_adapters_mvit(model, chain_type, config: AdapterConfig, experts=(1, 1), args=None):
    
#     chain_cls = CHAIN_MAP[chain_type]

#     stages = ["stages.3.1.transformer", "stages.4.1.transformer"]
#     for si, name in enumerate(stages):
#         # for mname, mm in model.named_modules():
#         #     print('mname', mname)
            
#         blocks = dict(model.named_modules())[name]
#         n = len(blocks)
#         inject_idxs = select_layers(
#             num_layers=n,
#             mode=config.layer_mode,
#             layers=config.layer_indices,
#             every=config.every,
#             last_k=config.last_k
#         )
#         experts_per_block = experts[si]

#         for idx in inject_idxs:
#             block = blocks[idx]

#             # print('idx, block', idx, type(block))
            
#             # MLP
#             if config.place_on in ("mlp", "both"):
#                 # compress mlp size if needed
                
#                 # for mname, mm in block.named_modules():
#                 #     print('mname', mname)
#                 #     # break
#                 conv = block.mlp.fc2 # TODO # need to run grad layer for mvit as well
#                 adapters = chain_cls(
#                     in_channels=conv.out_channels,
#                     num_blocks=1,
#                     experts_per_block=[experts_per_block],
#                     args=args
#                 )
#                 compress_mvit_mlp(block, args)
                
#                 # print('type(block.mlp.fc2.conv)', type(block.mlp.fc2.conv))
#                 # for mname, mm in block.mlp.fc2.conv.named_modules():
#                 #     print('mname', mname)
#                 #     break
#                 block.mlp.fc2 = nn.Sequential(conv, adapters)

#             # Conv mixer
#             if config.place_on in ("conv", "both") and hasattr(block, "attn"):
#                 if hasattr(block.attn, "out_proj"): # TODO 
#                     conv = block.attn.out_proj
#                     adapters = chain_cls(
#                         in_channels=conv.out_channels,
#                         num_blocks=1,
#                         experts_per_block=[experts_per_block],
#                         args=args
#                     )
#                     # print('type(block.token_mixer.talking_head2)', type(block.token_mixer.talking_head2))
#                     # for mname, mm in block.token_mixer.talking_head2.named_modules():
#                     #     print('mname', mname)
#                     #     break
#                     block.attn.out_proj = nn.Sequential(conv, adapters)

#     return model

def inject_adapters_mvit(model, chain_type, config: AdapterConfig, experts=(1, 1), args=None):
    """
    Injects residual adapters inside MobileViT transformer blocks.
    Adapter is applied as: y = fc2(x);  return y + adapter(y)
    """
    chain_cls = CHAIN_MAP[chain_type]

    # MobileViT transformer blocks to target
    target_stages = ["stages.3.1.transformer", "stages.4.1.transformer"]

    for si, name in enumerate(target_stages):

        transformer = dict(model.named_modules())[name]   # ModuleList
        blocks = list(transformer)
        n = len(blocks)

        inject_idxs = select_layers(
            num_layers=n,
            mode=config.layer_mode,
            layers=config.layer_indices,
            every=config.every,
            last_k=config.last_k
        )
        experts_per_block = experts[si]

        for idx in inject_idxs:
            block = blocks[idx]

            # -------------------------------------------------------
            #  MLP Injection:   y = fc2(...); return y + adapter(y)
            # -------------------------------------------------------
            if config.place_on in ("mlp", "both"):
                conv = block.mlp.fc2

                adapters = chain_cls(
                    in_channels=conv.out_channels,
                    num_blocks=1,
                    experts_per_block=[experts_per_block],
                    args=args
                )

                class WrappedFC2(nn.Module):
                    def __init__(self, conv, adapter):
                        super().__init__()
                        self.conv = conv
                        self.adapter = adapter

                    def forward(self, x):
                        y = self.conv(x)
                        return y + self.adapter(y)

                block.mlp.fc2 = WrappedFC2(conv, adapters)

            # -------------------------------------------------------
            #  Attention Out-Projection Injection
            # -------------------------------------------------------
            if config.place_on in ("conv", "both"):
                conv = block.attn.out_proj

                adapters = chain_cls(
                    in_channels=conv.out_channels,
                    num_blocks=1,
                    experts_per_block=[experts_per_block],
                    args=args
                )

                class WrappedAttnProj(nn.Module):
                    def __init__(self, conv, adapter):
                        super().__init__()
                        self.conv = conv
                        self.adapter = adapter

                    def forward(self, x):
                        y = self.conv(x)
                        return y + self.adapter(y)

                block.attn.out_proj = WrappedAttnProj(conv, adapters)

    return model


    
def inject_adapters_for_model(model, model_name, adapter_cfg, unfreeze_layers, hp, args):
    
    if "efficientformer" in model_name: 
    
        model = inject_adapters_eformer(
            model=model,
            chain_type=hp["chain_type"],
            config=adapter_cfg,
            experts=(hp["B5_adapters"], hp["B6_adapters"]),
            args=args
        )
        unfreeze_layers = ["stem", "stages.2", "stages.3", "head", "adapter"]
    elif "mobilevit" in model_name:

        # your mobilevit injector
        model = inject_adapters_mvit(
            model=model,
            chain_type=hp["chain_type"],
            config=adapter_cfg,
            experts=(hp["B5_adapters"], hp["B6_adapters"]),
            args=args
        )
        unfreeze_layers = ["stem", "stages.3.1", "stages.4.1", "head", "adapter"]
    else:
        raise ValueError(model_name)

    if hp["freeze_backbone"]:
        for n, p in model.named_parameters():
            p.requires_grad = ("adapter" in n) or n.startswith("head") or ('stem' in n)
    else:
        model = freeze_all_except(model, unfreeze_layers)

        if args.adapter_start_epoch > 0:
            for name, param in model.named_parameters():
                # freeze adapters if explicitly requested to start training after adapter_start_epoch
                if ("adapter" in name): param.requires_grad = False 
        
    return model

# https://arikpoz.github.io/posts/2025-04-29-low-rank-factorization-in-pytorch-compressing-neural-networks-with-linear-algebra/
def compress_layer(layer, epsilon=0.10, low_rank_dim=None, device='cpu', has_groups=False):
    """
    Compresses a layer using SVD if the compression is beneficial.
    Args:
        layer (nn.Module): The layer to compress.
        epsilon (float): The energy threshold for compression.
    Returns:
        nn.Module: The compressed layer or the original layer if compression is not beneficial.
    """

    # handle Conv2d layers
    assert isinstance(layer, nn.Conv2d), "This function is only for nn.Conv2d layers"
    # get convolution weight 4d matrix, shape: [out_channels, in_channels, kH, kW]
    W = layer.weight.data.cpu()
    OC, IC, kH, kW = W.shape

    # reshape to 2d matrix, with shape: [OC, IC*kH*kW]
    W_flat = W.view(OC, -1)

    # run SVD on flat weight matrix        
    U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)

    # find rank that capture the asked energy (1-epsilon)
    energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    rank = torch.searchsorted(energy, 1 - epsilon).item() + 1
    if low_rank_dim is not None:
        rank = low_rank_dim

    # check that factorization actually reduces number of parameters
    old_size = W.numel()
    new_size = rank * (IC * kH * kW + OC)
    if new_size < old_size:
        # define low rank factorization from SVD and rank
        U_r = U[:, :rank] @ torch.diag(S[:rank])
        V_r = Vh[:rank, :]

        # define two convolutional layers to replace the original convolutional layer
        conv1 = nn.Conv2d(
            in_channels=IC,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        conv2 = nn.Conv2d(
            in_channels=rank,
            out_channels=OC,
            kernel_size=(kH, kW),
            stride=layer.stride,
            padding=layer.padding,
            bias=(layer.bias is not None),
            # groups=1 if not has_groups else OC
        )
        conv1.weight.data = V_r.view(rank, IC, kH, kW).to(device)
        conv2.weight.data = U_r.view(OC, rank, 1, 1).to(device)
        if layer.bias is not None:
            conv2.bias.data = layer.bias.data.to(device)
        return nn.Sequential(conv1, conv2)
    return layer