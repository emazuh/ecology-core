import yaml
from pathlib import Path
from itertools import product
from copy import deepcopy

TEMPLATE = Path("experiments/templates/layer_search_tiny.yaml")

DATASETS = ["birds_inat", "rare_species"]
ADAPTER_TYPES = ["simple", "bottleneck"]
CHAIN_TYPES = ["par_fixed", "seq_fixed", "par_input_dep"]

OUT_ROOT = Path("experiments/birds/layer_search")

with open(TEMPLATE) as f:
    base = yaml.safe_load(f)

for dataset in DATASETS:
    for adapter_type, chain_type in product(ADAPTER_TYPES, CHAIN_TYPES):
        cfg = deepcopy(base)
    
        cfg["name"] = f"birds_layer_search_{adapter_type}_{chain_type}"
        cfg["adapter"] = {
            "type": adapter_type,
            "chain_type": chain_type,
            "config": None,
        }

        out_dir = Path(f"experiments/{dataset}/layer_search/{adapter_type}")
        out_dir.mkdir(parents=True, exist_ok=True)
    
        out_file = out_dir / f"{chain_type}.yaml"
        with open(out_file, "w") as f:
            yaml.safe_dump(cfg, f)
    
        print(f"Wrote {out_file}")
