import torch
import torch.nn as nn

class EntropyCELoss(nn.Module):
    """
    Standard CrossEntropy + entropy penalty computed in model.
    (args.gates_ent_loss is written by the model forward).
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, args, include_entropy=True):
        loss = self.ce(logits, labels)

        if include_entropy and hasattr(args, "gates_ent_loss"):
            loss = loss + args.gates_ent_loss

        return loss
