def freeze_all_except(model, allowed_layers):
    for name, p in model.named_parameters():
        p.requires_grad = any(layer in name for layer in allowed_layers)
    return model
