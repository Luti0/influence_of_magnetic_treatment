import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import List, Optional
import pandas as pd
from typing import Dict, Union
from scripts.evaluate import evaluate_model
import os

def render_mpl_table(
    data: pd.DataFrame,
    col_width: float = 3.0,
    row_height: float = 0.6,
    font_size: int = 14,
    header_color: str = '#40466e',
    row_colors: List[str] = ['#f1f1f2', 'w'],
    edge_color: str = 'w',
    bbox: List[float] = [0, 0, 1, 1],
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
) -> None:
    '''
    Plots a table from a pandas DataFrame using matplotlib.
    '''
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def compare_cv_results(
    results: Dict[str, Dict[str, float]],
    title: str = "Cross-Validation Results",
    save_path_img: str = "img/cv_metrics.png",
    save_path_csv: str = "results/cv_metrics.csv",
    selected_metrics: list = ['MSE', 'MAE', 'RMSE']
) -> None:
     
    """
    Metrics into CSV + graph.
    """
     
    os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)

    df = pd.DataFrame(results).T  #

    print("\n📋 Metrics after cross validation:")
    print(df.round(6))
    df.to_csv(save_path_csv)
    print(f"\n📄 Metrics saved in: {save_path_csv}")

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(selected_metrics):
        values = df[metric].values
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path_img)
    plt.show()

    print(f"📊 Graph saved in: {save_path_img}")
