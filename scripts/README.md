### Running experiments

1. Generate yamls for the experiment task (e.g. `layer_search`) using a template and generator script such as
   
```bash
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@g3113 ecology-core]$ python scripts/generate_layer_search_yamls.py 
Wrote experiments/birds_inat/layer_search/simple/par_fixed.yaml
Wrote experiments/birds_inat/layer_search/simple/seq_fixed.yaml
Wrote experiments/birds_inat/layer_search/simple/par_input_dep.yaml
Wrote experiments/birds_inat/layer_search/bottleneck/par_fixed.yaml
Wrote experiments/birds_inat/layer_search/bottleneck/seq_fixed.yaml
Wrote experiments/birds_inat/layer_search/bottleneck/par_input_dep.yaml
Wrote experiments/rare_species/layer_search/simple/par_fixed.yaml
Wrote experiments/rare_species/layer_search/simple/seq_fixed.yaml
Wrote experiments/rare_species/layer_search/simple/par_input_dep.yaml
Wrote experiments/rare_species/layer_search/bottleneck/par_fixed.yaml
Wrote experiments/rare_species/layer_search/bottleneck/seq_fixed.yaml
Wrote experiments/rare_species/layer_search/bottleneck/par_input_dep.yaml
```

2. Use run_all script to run each of the generated experiments

```bash
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@g3113 ecology-core]$ bash scripts/run_all_layer_search.sh
```

3. Monitor experiments on wandb and aggregate results as needed with `summarize_` script

```bash
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@g3113 ecology-core]$ python scripts/summarize_layer_search.py
```