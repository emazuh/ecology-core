import os
import wandb
import argparse
import yaml
import optuna
from pathlib import Path
from datetime import datetime
import subprocess

from hpo.objective import objective_adapter_layers
from utilities.results import finalize_study_best_model


USER = os.environ.get("USER", "unknown")

# -------------------------
# helpers
# -------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def build_pruner(pruner_cfg):
    if pruner_cfg["type"] == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=pruner_cfg["percentile"],
            n_warmup_steps=pruner_cfg.get("warmup_steps", 0),
        )
    raise ValueError(f"Unknown pruner type: {pruner_cfg['type']}")


def make_run_name(task, dataset, model, adapter, fraction, seed):
    return f"{task}/{dataset}/{model}/{adapter}/{fraction}/seed{seed}"


# -------------------------
# runners
# -------------------------

def run_layer_search(exp_cfg, args):
    task = exp_cfg["task"]
    # adapter = exp_cfg["adapter"]["chain_type"]
    adapter_type = exp_cfg["adapter"]["type"]
    chain_type = exp_cfg["adapter"]["chain_type"]
    adapter_name = f"{adapter_type}_{chain_type}"
    git_commit = get_git_commit()
    run_count = 0

    for seed in exp_cfg.get("seeds", [0]):
        for ds in exp_cfg["datasets"]:
            for frac in ds["fractions"]:
                for model in exp_cfg["models"]:

                    run_name = make_run_name(
                        task=task,
                        dataset=ds["name"],
                        model=model,
                        adapter=adapter_name,
                        fraction=frac,
                        seed=seed,
                    )

                    if args.dry_run:
                        print(run_name)
                        run_count += 1
                        continue


                    wandb_run = None
                    if exp_cfg["logging"]["wandb"] and not args.dry_run:
                        wandb_run = wandb.init(
                            project=exp_cfg["logging"]["project"],
                            name=run_name,
                            group=exp_cfg["name"],        # 🔑 experiment-level
                            job_type=exp_cfg["task"],
                            tags=[
                                "cvpr25",
                                exp_cfg["task"],          # layer_search
                                ds["name"],               # birds_inat
                                model,                    # mobilevit
                                adapter_type,        # bottleneck
                                chain_type,          # par_fixed
                                f"{int(frac * 100)}p",    # 5p, 10p, 25p
                                USER,
                            ],
                            config={
                                "experiment": exp_cfg["name"],
                                "task": exp_cfg["task"],
                                "dataset": ds["name"],
                                "model": model,
                                "fraction": frac,
                                "adapter_type": adapter_type,
                                "chain_type": chain_type,
                                "seed": seed,
                                "epochs": exp_cfg["hpo"]["epochs"],
                            },
                            reinit=True,
                        )

                    study = optuna.create_study(
                        study_name=run_name,
                        direction="maximize",
                        pruner=build_pruner(exp_cfg["hpo"]["pruner"]),
                    )

                    study.optimize(
                        lambda trial: objective_adapter_layers(
                            trial,
                            adapter_cfg=exp_cfg["adapter"]["config"],
                            adapter_type=exp_cfg["adapter"].get("type"),
                            epochs=exp_cfg["hpo"]["epochs"],
                            model_name=model,
                            dataset_name=ds["name"],
                            data_subset=frac,
                            chain_type=chain_type,
                            log_wandb=exp_cfg["logging"]["wandb"],
                            wandb_project=exp_cfg["logging"]["project"],
                            seed=seed,
                            single_objective=True,
                        ),
                        n_trials=exp_cfg["hpo"]["n_trials"],
                    )

                    if wandb_run is not None:
                        wandb_run.summary["best_val_acc"] = study.best_value
                        wandb.finish()

                    finalize_study_best_model(
                        study,
                        output_dir=Path("results") / task,
                        metadata={
                            "experiment": exp_cfg["name"],
                            "run_name": run_name,
                            "git_commit": git_commit,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

    if args.dry_run:
        print()
        print(f"✓ {run_count} runs generated")
        print(f"✓ task: {task}")
        print(f"✓ experiment: {exp_cfg['name']}")


# -------------------------
# entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiment)

    if exp_cfg["task"] == "layer_search":
        run_layer_search(exp_cfg, args)
    else:
        raise ValueError(f"Unknown task: {exp_cfg['task']}")


if __name__ == "__main__":
    main()
