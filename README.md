The project structure is as below. Main results is generated from `cd scripts && python generate_main_table.py`.
The easiest dataset to try is `rare_species` since it's on [huggingface](https://huggingface.co/datasets/imageomics/rare-species).
iWildCam is also downloadable via scripts.

```
├── dataloaders
│   ├── dataloaders.py
│   └── data
├── adapters
│   ├── injector.py
│   ├── freeze.py
│   ├── chains.py
│   └── registry.py
├── hpo
│   ├── outputs
│   ├── layers_objective.py
│   ├── objective.py
│   └── mvit_birds_general_best.txt
├── notebooks
│   ├── birds
│   ├── measurement
│   ├── MoE-GradCAM.ipynb
│   ├── GradCAM.ipynb
│   ├── AnalysisMain.ipynb
│   └── wilds
├── models
│   ├── models_utils.py
│   └── __init__.py
├── configs
│   ├── general_config.json
│   ├── search_space.py
│   └── adapter_config.py
├── utilities
│   └── wandb_run_best.py
├── scripts
│   ├── run_mvit_layer_search.py
│   ├── MainTable.ipynb
│   ├── run_eformer_layer_search.py
│   ├── adapter_layers
│   ├── ckpts
│   ├── generate_main_table.py
└── training
    ├── engine.py
    ├── scheduler_utils.py
    ├── logging.py
    ├── utils.py
    ├── grad_norm_utils.py
    └── losses.py
```


## Experiment Boundaries (IMPORTANT)

- Training logic lives in `training/`, `models/`, `hpo/`
- Students must not modify these directories
- New experiments should be expressed via:
  - notebooks (analysis only)
  - scripts (wrappers only)
- Reproducibility is enforced via `utilities/reproducibility.py`
- Minor reproducibility test `python -m utilities.test_reproducibility`
