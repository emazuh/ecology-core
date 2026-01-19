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

## Setting up the conda environment

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p /gscratch/stf/$USER/miniconda3
rm -rf ~/miniconda3/miniconda.sh
# you might have to close and reopen the terminal for conda command to be available
conda create --prefix=/gscratch/scrubbed/$USER/ecologyenv python=3.10
conda config --append envs_dirs /gscratch/scrubbed/$USER
mkdir /gscratch/scrubbed/$USER
cd /gscratch/scrubbed/$USER
git clone git@github.com:emazuh/ecology-core.git
cd ecology-core
[emazuh@klone-login01 ecology-core]$ conda activate ecologyenv
[emazuh@klone-login01 ecology-core]$ export PYTHONUTF8=1
[emazuh@klone-login01 ecology-core]$ export PYTHONIOENCODING=utf-8
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$
pip install -r requirements.txt
```

## Experiment Boundaries (IMPORTANT)

- Training logic lives in `training/`, `models/`, `hpo/`
- Students must not modify these directories
- New experiments should be expressed via:
  - notebooks (analysis only)
  - scripts (wrappers only)
- Reproducibility is enforced via `utilities/reproducibility.py`
- Minor reproducibility test `python -m utilities.test_reproducibility`
