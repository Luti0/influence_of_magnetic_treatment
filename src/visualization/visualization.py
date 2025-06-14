import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import List, Optional
import pandas as pd
from typing import Dict, Union
# from evaluation.evaluate import evaluate_model
import os

def render_mpl_table(data: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(data.shape[1] * 2.5, data.shape[0] * 0.6 + 1))
    ax.axis('off')

    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='w')
        else:
            cell.set_facecolor('#f1f1f2' if row % 2 == 0 else 'w')
        cell.set_edgecolor('w')

    plt.show()


def compare_cv_results(
    results: Dict[str, Dict[str, float]],
    title: str = "Cross-Validation Results",
    save_path_csv: str = "results/cv_metrics.csv",
    selected_metrics: list = ['MSE', 'MAE', 'RMSE', 'R²'],
    display_plots: bool = True
) -> None:
    """
    Creates plot for each metric.
    """
    
    os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)

    df = pd.DataFrame(results).T

    selected_metrics = [m for m in selected_metrics if m in df.columns]
    if not selected_metrics:
        raise ValueError("None of the selected metrics are present in the results.")

    print("\n📋 Metrics after cross validation:")
    print(df.round(6))
    
    df.to_csv(save_path_csv)
    print(f"\n📄 Metrics saved in: {save_path_csv}")


    x = np.arange(len(df))
    for metric in selected_metrics:

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        values = df[metric].values
        ax.bar(x, values, label=metric, color='#40466e', alpha=1)

        ax.set_ylabel(f"{metric} value")
        ax.set_title(f"{title} - {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45)
        ax.legend()

        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        if display_plots:
            plt.show()

        plt.close(fig)