import wandb
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


class WandbRunFetcher:
    def __init__(self, entity_project: str, filters: Dict[str, Any]):
        self.api = wandb.Api()
        self.entity_project = entity_project
        self.filters = filters

    def fetch(self):
        return self.api.runs(self.entity_project, filters=self.filters)


class SubmodelGrouper:
    def __init__(self, submodel_keys: List[str], eval_key:str = "test_acc", extra_fields:list = []):
        """
        submodel_keys: list of config keys that define a submodel.
        """
        self.submodel_keys = submodel_keys
        self.eval_key = eval_key
        self.extra_fields = extra_fields

    def extract_submodel_key(self, config: Dict[str, Any]) -> Tuple:
        """
        Returns an immutable tuple representation of the submodel.
        Missing keys become None.
        """
        submodel_cfg = {k: config.get(k, None) for k in self.submodel_keys}
        return tuple(submodel_cfg.items())

    def build_records(self, runs):
        records = []

        for run in runs:
            cfg = run.config
            summary = run.summary

            acc = summary.get(self.eval_key)
            if acc is None:
                continue

            submodel_key = self.extract_submodel_key(cfg)

            record = {
                "run_id": run.id,
                self.eval_key: float(acc),
                "submodel_key": submodel_key,
            }

            # Add the submodel columns explicitly for readability
            for k in self.submodel_keys:
                record[k] = cfg.get(k, None)

             # Include extra requested fields
            for field in self.extra_fields:
                record[field] = cfg.get(field, summary.get(field, None))
                
            records.append(record)

        return pd.DataFrame(records)


class BestRunSelector:
    @staticmethod
    def top_n_per_submodel(df: pd.DataFrame, n: int = 3, eval_key:str = "test_acc"):
        """
        Returns the top-n runs for each submodel_key.
        """
        return (
            df.sort_values(eval_key, ascending=False)
              .groupby("submodel_key")
              .head(n)
              .reset_index(drop=True)
        )

    @staticmethod
    def best_unique_submodels(df: pd.DataFrame, top_k: int = 3, eval_key:str = "test_acc"):
        """
        Select the best run (highest acc) from each submodel,
        then return the best K submodels.
        """
        best = (
            df.sort_values(eval_key, ascending=False)
              .groupby("submodel_key")
              .head(1)
              .reset_index(drop=True)
        )
        return best.head(top_k)


class OptunaSubmodelAnalyzer:
    def __init__(self, project_path: str, filters: Dict[str, Any], submodel_keys: List[str], eval_key:str = "test_acc",
                extra_fields:list = []):
        self.fetcher = WandbRunFetcher(project_path, filters)
        self.grouper = SubmodelGrouper(submodel_keys, eval_key=eval_key, extra_fields=extra_fields)
        self.selector = BestRunSelector()
        self.eval_key = eval_key

    def load_dataframe(self) -> pd.DataFrame:
        runs = self.fetcher.fetch()
        return self.grouper.build_records(runs)

    def get_topk_per_submodel(self, k=3) -> pd.DataFrame:
        df = self.load_dataframe()
        return self.selector.top_n_per_submodel(df, n=k, eval_key=self.eval_key)

    def get_topk_submodels(self, k=3) -> pd.DataFrame:
        df = self.load_dataframe()
        return self.selector.best_unique_submodels(df, top_k=k, eval_key=self.eval_key)
