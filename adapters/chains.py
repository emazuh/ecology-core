# chain classes
# class ChainSequential: ...
# class ChainSequentialInputDependent: ...
# class ChainParallelFixed: ...
# class ChainParallelInputDependent: ...

import math
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F 

from collections import defaultdict
import torch
import numpy as np

SCALES = 0.001 # 0.1
# -------------------------
# Conv Adapter
# -------------------------
class ConvAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)


class ConvAdapterBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1, reduction=16, adapter_type="bottleneck"):
        """
        adapter_type: "simple" or "bottleneck"
        """
        super().__init__()
        padding = kernel_size // 2
        out_channels = out_channels or in_channels

        # Bottleneck adapter
        reduction = reduction if reduction else 16
        hidden = max(1, in_channels // reduction)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(1, in_channels)
        )

    def forward(self, x):
        return self.adapter(x)

def gate_entropy(gates):
    ent = 0.0
    for g in gates:
        if g is not None:
            p = torch.sigmoid(g)
            ent += -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).sum()
    return ent

# class GlobalRFFExpert(nn.Module):
#     def __init__(self, c_in, D=64, gamma=0.5):
#         super().__init__()
#         self.D = D
#         self.W = nn.Parameter(torch.randn(D, c_in) * gamma, requires_grad=False)
#         self.b = nn.Parameter(2 * torch.pi * torch.rand(D), requires_grad=False)

#         # Project back to channels (pointwise)
#         self.linear = nn.Linear(D, c_in)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         B, C, H, W = x.shape

#         # global avg pool: (B, C)
#         g = x.mean(dim=[2, 3])             # (B, C)

#         # RFF on global vector: (B, D)
#         proj = torch.cos(g @ self.W.T + self.b) * (2.0 ** 0.5)

#         # Mix back to channels: (B, C)
#         g_out = self.linear(proj)

#         # broadcast to spatial size
#         g_out = g_out[:, :, None, None]    # (B, C, 1, 1)
#         g_out = g_out.expand(-1, -1, H, W) # (B, C, H, W)

#         return g_out

# class GlobalRFFExpert(nn.Module):
#     def __init__(self, c_in, D=64, gamma=0.5):
#         super().__init__()
#         C = c_in
#         self.D = D

#         self.W = nn.Parameter(torch.randn(D, C) * gamma, requires_grad=False)
#         self.b = nn.Parameter(2 * torch.pi * torch.rand(D), requires_grad=False)

#         self.linear = nn.Conv2d(D, C, 1, bias=False)
#         self.bn = nn.BatchNorm2d(C)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # RFF per pixel
#         x_flat = x.permute(0,2,3,1).reshape(-1, C)
#         proj = torch.cos(x_flat @ self.W.T + self.b) / math.sqrt(self.D)

#         proj = proj.reshape(B, H, W, self.D).permute(0,3,1,2)

#         out = self.linear(proj)
#         out = self.bn(out)
#         out = self.act(out)
#         return out


class GlobalRFFExpert(nn.Module):
    def __init__(self, c_in, D=8, r=16, gamma=0.5): # D=64
        super().__init__()
        C = c_in
        self.D = D
        self.r = r

        # Fixed RFF parameters (freeze for ANE)
        self.W = nn.Parameter(torch.randn(D, C) * gamma, requires_grad=False)
        self.b = nn.Parameter(2 * torch.pi * torch.rand(D), requires_grad=False)

        # Low-rank projection using 1x1 convs
        self.reduce = nn.Conv2d(D, r, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(r, C, kernel_size=1, bias=False)

        # Standard BN+ReLU for adapter semantics
        self.bn = nn.BatchNorm2d(C)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply RFF per pixel
        x_flat = x.permute(0,2,3,1).reshape(-1, C)
        proj = torch.cos(x_flat @ self.W.T + self.b) * (1.0 / math.sqrt(self.D))
        proj = proj.reshape(B, H, W, self.D).permute(0,3,1,2)

        # Low-rank 1x1 projections
        h = self.reduce(proj)
        h = self.expand(h)

        # Standard adapter finish
        h = self.bn(h)
        h = self.act(h)
        return h



def get_block_experts(in_channels, experts_per_block, bidx, args):
    if args.adapter_type == "simple":
        print('[INFO]: Using conv adapter experts')
        block_experts = nn.ModuleList([ConvAdapter(in_channels, in_channels) for _ in range(experts_per_block[bidx])])
    elif args.adapter_type == "bottleneck":
        print('[INFO]: Using bottleneck adapter experts')
        block_experts = nn.ModuleList([ConvAdapterBottleneck(in_channels, in_channels, 
                                                             reduction=args.reduction) for _ in range(experts_per_block[bidx])])
    elif args.adapter_type == "fourier":
        print('[INFO]: Using fourier adapter experts')
        block_experts = nn.ModuleList([GlobalRFFExpert(c_in=in_channels, D=8) for _ in range(experts_per_block[bidx])]) # 8
    else:
        raise ValueError(f"Unknown adapter_type: {args.adapter_type}")

    return block_experts
    
# -------------------------
# Variant 1: Sequential chain
# -------------------------
class ChainSequential(nn.Module):
    def __init__(self, in_channels, num_blocks=1, experts_per_block=None, args=None):
        super().__init__()
        if experts_per_block is None:
            experts_per_block = [1]*num_blocks
        self.blocks = nn.ModuleList()
        self.gates = nn.ParameterList()
        self.scales = nn.ParameterList()
        print('[INFO]: Using ChainSequential')
        if args is not None: 
            # No extra dictionaries when deploying
            if (not hasattr(args, 'deploy')) or (hasattr(args, 'deploy') and not args.deploy):
                self._args = args
        for b in range(num_blocks):
            # if args.adapter_type == "simple":
            #     print('[INFO]: Using conv adapter experts')
            #     block_experts = nn.ModuleList([ConvAdapter(in_channels, in_channels) for _ in range(experts_per_block[b])])
            # elif args.adapter_type == "fourier":
            #     print('[INFO]: Using fourier adapter experts')
            #     block_experts = nn.ModuleList([GlobalRFFExpert(in_channels, in_channels) for _ in range(experts_per_block[b])])
            # else:
            #     raise ValueError(f"Unknown adapter_type: {args.adapter_type}")
            block_experts = get_block_experts(in_channels, experts_per_block, b, args)
            self.blocks.append(block_experts)
            self.gates.append(nn.Parameter(torch.ones(experts_per_block[b])) if experts_per_block[b] > 0 else None)
            self.scales.append(nn.Parameter(torch.tensor(SCALES)))

    def forward(self, x):
        out = x
        for b_idx, block in enumerate(self.blocks):
            block_out = 0
            for e_idx, expert in enumerate(block):
                gate = torch.sigmoid(self.gates[b_idx][e_idx])
                block_out = block_out + gate * expert(out)
            out = out + self.scales[b_idx] * block_out

            if hasattr(self, "_args") and hasattr(self._args, "gate_logger") and self._args.gate_logger:
                self._args.gate_logger.record(
                    block_idx=b_idx,
                    gates=torch.sigmoid(self.gates[b_idx]),
                    labels=getattr(self._args.gate_logger, "current_labels", None)
                )
            
        if hasattr(self, '_args'):
            ent_value = getattr(self._args, 'entropy_coeff', 1e-4) * gate_entropy(self.gates)
            if not hasattr(self._args, 'gates_ent_loss'): self._args.gates_ent_loss = 0.0 # torch.tensor(0)
            self._args.gates_ent_loss -= ent_value # maximize entropy

            if not hasattr(self._args, 'gate_values'):
                self._args.gate_values = []
            # Log current sigmoid gate activations
            # print('gates', self.gates)
            current_gates = [torch.sigmoid(g.detach()) for g in self.gates if g is not None] # g.detach()).mean()
            self._args.gate_values.append(current_gates)
        
        return out


# -------------------------
# Variant 2: Parallel experts, fixed gating
# -------------------------
class ChainParallelFixed(nn.Module):
    def __init__(self, in_channels, num_blocks=1, experts_per_block=None, args=None):
        super().__init__()
        if experts_per_block is None:
            experts_per_block = [1]*num_blocks
        self.blocks = nn.ModuleList()
        self.gates = nn.ParameterList()
        self.scales = nn.ParameterList()
        if args is not None: 
            # No extra dictionaries when deploying
            if (not hasattr(args, 'deploy')) or (hasattr(args, 'deploy') and not args.deploy):
                self._args = args
        print('[INFO]: Using ChainParallelFixed')
        for b in range(num_blocks):
            block_experts = get_block_experts(in_channels, experts_per_block, b, args)
            self.blocks.append(block_experts)
            self.gates.append(nn.Parameter(torch.ones(experts_per_block[b])) if experts_per_block[b] > 0 else None)
            self.scales.append(nn.Parameter(torch.tensor(SCALES))) # 0.1

    def forward(self, x):
        out = x
        for b_idx, block in enumerate(self.blocks):
            if len(block) == 0:
                continue
            gates = torch.sigmoid(self.gates[b_idx])
#             gates = torch.clamp(F.relu(self.gates[b_idx]), max=1.0)

            gates_reshaped = gates.view(1, len(block), 1, 1, 1)
            # compute expert outputs and stack along new dimension
            expert_outs = torch.stack([expert(out) for expert in block], dim=1)  # [B, E, C, H, W]
            block_out = (expert_outs * gates_reshaped).sum(dim=1)
    
            # block_out = sum(expert(out) * gates[e_idx] for e_idx, expert in enumerate(block))
            # out = out + block_out
            out = out + self.scales[b_idx] * block_out

            if hasattr(self, "_args") and hasattr(self._args, "gate_logger") and self._args.gate_logger:
                self._args.gate_logger.record(
                    block_idx=b_idx,
                    gates=gates,
                    labels=getattr(self._args.gate_logger, "current_labels", None)
                )

            
        if hasattr(self, '_args'):
            ent_value = getattr(self._args, 'entropy_coeff', 1e-4) * gate_entropy(self.gates)
            # print('self.gates', self.gates[0])
            if not hasattr(self._args, 'gates_ent_loss'): self._args.gates_ent_loss = 0.0 # torch.tensor(0)
            self._args.gates_ent_loss -= ent_value # maximize entropy
#             print('ent_value', ent_value, gates)
            
#             for i, g in enumerate(self.gates):
#                 if g.grad is not None:
#                     print(f"Block {i} gate grad:", g.grad)
        
        return out

                
# -------------------------
# Variant 3: Parallel experts, input-dependent gating
# -------------------------
class ChainParallelInputDependent(nn.Module):
    def __init__(self, in_channels, num_blocks=1, experts_per_block=None, args=None):
        super().__init__()
        if experts_per_block is None:
            experts_per_block = [1]*num_blocks
        self.blocks = nn.ModuleList()
        self.adapter_gates = nn.ModuleList()
        self.scales = nn.ParameterList()
        if args is not None: 
            # No extra dictionaries when deploying
            if (not hasattr(args, 'deploy')) or (hasattr(args, 'deploy') and not args.deploy):
                self._args = args
        print('[INFO]: Using ChainParallelInputDependent')
        for b in range(num_blocks):
            # block_experts = nn.ModuleList([ConvAdapter(in_channels, in_channels) for _ in range(experts_per_block[b])])
            block_experts = get_block_experts(in_channels, experts_per_block, b, args)
            self.blocks.append(block_experts)
            if experts_per_block[b] > 0:
                self.adapter_gates.append(nn.Linear(in_channels, experts_per_block[b]))
            else:
                self.adapter_gates.append(None)
            self.scales.append(nn.Parameter(torch.tensor(SCALES)))

    def forward(self, x):
        out = x
        B, C, H, W = out.shape
        pooled = out.mean(dim=[2,3])  # Global pool
        for b_idx, block in enumerate(self.blocks):
            if len(block) == 0:
                continue
            
            if self.adapter_gates[b_idx] is not None:
                # Compute per-input gates
                gates = torch.sigmoid(self.adapter_gates[b_idx](pooled))  # [B, E]
                
                if hasattr(self, '_args'):
                    # Compute entropy per batch
                    p = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
                    batch_ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()                
                    ent_value = getattr(self._args, 'entropy_coeff', 1e-4) * batch_ent
                    if not hasattr(self._args, 'gates_ent_loss'): self._args.gates_ent_loss = 0.0 # torch.tensor(0)
                    self._args.gates_ent_loss -= ent_value # maximize entropy

                    if hasattr(self._args, "gate_logger") and self._args.gate_logger:
                        self._args.gate_logger.record(
                            block_idx=b_idx,
                            gates=gates,
                            labels=getattr(self._args.gate_logger, "current_labels", None)
                        )
                
                gates = gates.view(B, len(block), 1, 1, 1)  # [B, E, 1, 1, 1] for broadcasting
                # Compute all expert outputs and stack
                expert_outs = torch.stack([expert(out) for expert in block], dim=1)  # [B, E, C, H, W]
                # Weighted sum along expert dimension
                block_out = (expert_outs * gates).sum(dim=1)  # [B, C, H, W]
            else:
                # No gate → simple sum
                block_out = sum(expert(out) for expert in block)
            
            out = out + self.scales[b_idx] * block_out
        return out


class ChainSequentialInputDependent(nn.Module):
    def __init__(self, in_channels, num_blocks=1, experts_per_block=None, args=None):
        super().__init__()
        if experts_per_block is None:
            experts_per_block = [1] * num_blocks

        self.blocks = nn.ModuleList()
        self.adapter_gates = nn.ModuleList()
        self.scales = nn.ParameterList()
        if args is not None: 
            # No extra dictionaries when deploying
            if (not hasattr(args, 'deploy')) or (hasattr(args, 'deploy') and not args.deploy):
                self._args = args
        print('[INFO]: Using ChainSequentialInputDependent')
        for b in range(num_blocks):
            # Each block has multiple experts
            # block_experts = nn.ModuleList([
            #     ConvAdapter(in_channels, in_channels) 
            #     for _ in range(experts_per_block[b])
            # ])
            block_experts = get_block_experts(in_channels, experts_per_block, b, args)
            self.blocks.append(block_experts)
            self.scales.append(nn.Parameter(torch.tensor(SCALES)))

            # Add an input-dependent gate (per block)
            if experts_per_block[b] > 0:
                self.adapter_gates.append(nn.Linear(in_channels, experts_per_block[b]))
            else:
                self.adapter_gates.append(None)

    def forward(self, x):
        out = x
        B, C, H, W = out.shape

        for b_idx, block in enumerate(self.blocks):
            if len(block) == 0:
                continue

            # Compute pooled representation for gating
            pooled = out.mean(dim=[2, 3])  # shape: [B, C]

            # Compute gates based on input features
            if self.adapter_gates[b_idx] is not None:
                gates = torch.sigmoid(self.adapter_gates[b_idx](pooled))  # [B, num_experts]
                
                if hasattr(self, '_args'):
                    # Compute entropy per batch
                    p = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
                    batch_ent = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()                
                    ent_value = getattr(self._args, 'entropy_coeff', 1e-4) * batch_ent
                    if not hasattr(self._args, 'gates_ent_loss'): self._args.gates_ent_loss = 0.0 # torch.tensor(0)
                    self._args.gates_ent_loss -= ent_value # maximize entropy

                    if hasattr(self._args, "gate_logger") and self._args.gate_logger:
                        self._args.gate_logger.record(
                            block_idx=b_idx,
                            gates=gates,
                            labels=getattr(self._args.gate_logger, "current_labels", None)
                        )
                    
                gates = gates.unsqueeze(-1).unsqueeze(-1)  # [B, num_experts, 1, 1]
            else:
                gates = torch.ones(B, len(block), 1, 1, device=x.device)

            # Combine experts weighted by gates (per-sample, per-expert)
            block_out = 0
            for e_idx, expert in enumerate(block):
                block_out = block_out + self.scales[b_idx] * gates[:, e_idx].view(B, 1, 1, 1) * expert(out)

            # Sequential residual update
            out = out + block_out

        return out