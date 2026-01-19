from types import SimpleNamespace
import torch
from timm.scheduler import create_scheduler

def build_optimizer(model, lr, wd):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd
    )

def build_scheduler(args, optimizer):
    """
    Returns timm scheduler + epoch length.
    """
    # sched_cfg = dict(
    #     sched='cosine',
    #     epochs=args.epochs,
    #     min_lr=0.0,
    #     warmup_lr=1e-5,
    #     warmup_epochs=5,
    # )
    # scheduler, _ = create_scheduler(
    #     type("Dummy", (), sched_cfg),
    #     optimizer
    # )
    scheduler, _ = create_scheduler(SimpleNamespace(**{ 'sched': 'cosine', 'epochs': args.epochs,
                                                        'min_lr': 0.0, 'warmup_lr': 1e-5, 'warmup_epochs': 5, }), optimizer)
    return scheduler

def update_scheduler_per_step(scheduler, epoch, iter_per_epoch, batch_idx, args):
    if not scheduler:
        return
        
    num_updates = epoch # epoch * iter_per_epoch + batch_idx
    if hasattr(scheduler, "step_update"):
        scheduler.step_update(num_updates)
    else:
        scheduler.step(num_updates)
