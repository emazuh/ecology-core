import optuna

def suggest_stage1_hyperparameters(trial, epochs):    
    hp = {}

    # Fixed or dynamic params
    hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
    hp["randerase_p"] = trial.suggest_float("randerase_p", 0.0, 0.5)
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    return hp


def suggest_stage2_hyperparameters(trial, epochs, freeze_backbone=None, adapter_start_epoch=None):
    hp = {}

    # Fixed or dynamic params
    hp["lr"] = 0.0008
    hp["batch_size"] = 256
    hp["dropout"] = 0.25
    hp["randerase_p"] = 0.36

    # Adapter fan-out
    hp["B5_adapters"] = trial.suggest_categorical("B5_adapters", [1, 2, 4, 8])
    hp["B6_adapters"] = trial.suggest_categorical("B6_adapters", [1, 2, 4, 8])

    hp["place_on"] = trial.suggest_categorical("place_on", ["mlp", "conv"])
    hp["layer_mode"] = trial.suggest_categorical("layer_mode", ["every", "all"])

    # Backbone freeze
    if freeze_backbone is None:
        hp["freeze_backbone"] = trial.suggest_categorical("freeze_backbone", [True, False])
    else:
        hp["freeze_backbone"] = freeze_backbone

    # Gating
    if hp["freeze_backbone"] is False:
        hp["input_dependent_gating"] = trial.suggest_categorical(
            "input_dependent_gating", [True, False]
        )
    else:
        hp["input_dependent_gating"] = False

    # Adapter chain type
    hp["chain_type"] = trial.suggest_categorical(
        "chain_type", ["seq", "seq_input", "par_fixed", "par_input"]
    )

    # Entropy loss coeff
    hp["entropy_coeff"] = trial.suggest_categorical(
        "entropy_coeff", [-1e-2, 1e-2, 0.1, -0.1, 1e-3, -1e-3]
    )

    # Update start-epoch
    if adapter_start_epoch is None:
        hp["adapter_start_epoch"] = trial.suggest_int(
            "adapter_start_epoch",
            0 if hp["freeze_backbone"] else 1,
            epochs // 2
        )
    else:
        hp["adapter_start_epoch"] = adapter_start_epoch

    # Adapter size/scaling
    hp["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", [0.1, 1.0, 5.0])
    hp["adapter_scale"] = trial.suggest_categorical("adapter_scale", [0.1, 1.0, 2.0, 10.0])
    hp["adapter_type"] = trial.suggest_categorical("adapter_type", ["simple", "fourier"]) # "bottleneck"

    if hp["adapter_type"] == "bottleneck":
        hp["reduction"] = trial.suggest_categorical("reduction", [4, 8, 16, 32])
    else:
        hp["reduction"] = None

    return hp
