import pandas as pd
from pathlib import Path
import pickle

"""
For asking questions like:

Which adapter types are clearly worse? → drop them

Which chain types are unstable? → don’t scale them

Which model + adapter combo dominates? → focus
"""

rows = []

for pkl in Path("results/layer_search").glob("*.pkl"):
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    rows.append({
        "run": pkl.stem,
        "best_val_acc": data["best_trial"]["value"],
        **data["config"],
    })

df = pd.DataFrame(rows)
print(df.sort_values("best_val_acc", ascending=False))
