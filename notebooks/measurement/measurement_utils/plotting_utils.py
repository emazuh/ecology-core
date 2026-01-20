import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Compute Pareto frontier
def is_pareto_efficient(points, maximize=(True, False)):
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[i] = not np.any([
                all((p[j] >= c[j] if maximize[j] else p[j] <= c[j]) for j in range(len(c))) and
                any((p[j] > c[j] if maximize[j] else p[j] < c[j]) for j in range(len(c)))
                for p in points
            ])
    return is_efficient
    
def plot_pareto_frontier(frontier_dataframe, save_loc, font_size=13, add_title=True):
    frontier_dataframe = frontier_dataframe.copy()
    points = frontier_dataframe[["accuracy", "latency"]].values
    frontier_dataframe.loc[:, "is_pareto"] = is_pareto_efficient(points, maximize=(True, False))

    # Sort by model and latency for proper line connections
    frontier_dataframe_sorted = frontier_dataframe.sort_values(by=["model_name", "latency"])

    plt.figure(figsize=(12, 8))

    # Plot lines connecting configurations of the same model
    sns.lineplot(
        data=frontier_dataframe_sorted,
        x="latency",
        y="accuracy",
        hue="model_name",
        style="model_name",
        markers=True,
        dashes=True,
        palette="Set2",
        linewidth=1.8,
        markersize=10,
        legend="brief"
    )

    # Highlight Pareto points
    pareto_df = frontier_dataframe_sorted[frontier_dataframe_sorted["is_pareto"]]
    sns.scatterplot(
        data=pareto_df,
        x="latency",
        y="accuracy",
        s=250,   # Pareto points larger
        color="red",
        label="Pareto frontier",
        zorder=10
    )

    from adjustText import adjust_text

    texts = []
    for _, row in frontier_dataframe_sorted.iterrows():
        t = plt.text(
            row["latency"],
            row["accuracy"],
            f"{row['model_name']}\n{row['config']}",
            fontsize=font_size,
            ha="left",
            va="bottom",
        )
        texts.append(t)
    
    # Auto-adjust positions & add connecting lines
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        autoalign=True,
        only_move={'points':'y', 'text':'xy'},
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4)
    )

    plt.xscale("log")
    plt.xlabel("Latency (ms) [log scale]", fontsize=font_size)
    plt.xticks(fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if add_title:
        plt.title("Pareto Frontier: Accuracy vs Latency")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.legend(fontsize=font_size)
    plt.savefig(save_loc)
    plt.show()