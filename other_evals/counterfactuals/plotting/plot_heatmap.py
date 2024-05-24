from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_csv_and_plot_heatmap(csv_path: str | Path) -> None:
    """
    Load a CSV file and plot a heatmap with the mean values and 95% confidence intervals.

    Parameters:
    - csv_path: str, the path to the CSV file.
    """
    data = pd.read_csv(csv_path)
    assert len(data) > 0, "The CSV file is empty."
    eval_name: str = Path(csv_path).stem
    plot_heatmap_with_ci(data, title=f"{eval_name} Percentage of Meta Predicted Correctly with 95% CI")


def plot_heatmap_with_ci(
    data,
    value_col: str = "meta_predicted_correctly",
    object_col: str = "object_model",
    meta_col: str = "meta_model",
    title: str = "Percentage of Meta Predicted Correctly with 95% CI",
):
    """
    Plots a heatmap with the mean values and 95% confidence intervals.

    Parameters:
    - data: pd.DataFrame, the input data containing the necessary columns.
    - value_col: str, the column name for the values to calculate mean and CI.
    - object_col: str, the column name for the object models.
    - meta_col: str, the column name for the meta models.
    - title: str, the title of the heatmap.
    """
    # Calculate the mean and 95% confidence interval for each combination
    grouped_data = data.groupby([object_col, meta_col])[value_col].agg(["mean", "count", "std"])
    grouped_data["sem"] = grouped_data["std"] / np.sqrt(grouped_data["count"])
    grouped_data["95%_ci"] = grouped_data["sem"] * 1.96

    # Prepare data for heatmap with 95% CI annotations
    heatmap_data = grouped_data["mean"].unstack().T * 100
    ci_data = grouped_data["95%_ci"].unstack().T * 100
    number_samples = grouped_data["count"].unstack().T

    # Create a seaborn heatmap with annotations for 95% CI
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", cbar=False)

    # Annotate the heatmap with 95% CI slightly below the main percentage value
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            mean_value = heatmap_data.iloc[i, j]
            ci_value = ci_data.iloc[i, j]
            lower_bound = mean_value - ci_value
            upper_bound = mean_value + ci_value
            n = number_samples.iloc[i, j]
            ax.text(
                j + 0.5,
                i + 0.75,
                f"({lower_bound:.1f} - {upper_bound:.1f}), n={n}",
                color="white",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.title(title)
    plt.xlabel("Object Model")
    plt.ylabel("Meta Model")
    plt.show()
