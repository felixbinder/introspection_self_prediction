from typing import List, Sequence, Tuple

import pandas as pd
from pydantic import BaseModel


def pandas_equal_frequency_binning(
    data: List[Tuple[float, float]], num_bins: int = 10
) -> Tuple[List[float], List[float]]:
    df = pd.DataFrame(data, columns=["expected_prob", "predicted_prob"])
    df["bin"] = pd.qcut(df["expected_prob"], q=num_bins, duplicates="drop")

    bin_means = df.groupby("bin", observed=False).agg({"expected_prob": "mean", "predicted_prob": "mean"}).reset_index()

    return bin_means["expected_prob"].tolist(), bin_means["predicted_prob"].tolist()


class CalibrationData(BaseModel):
    expected_prob: float
    predicted_prob: float
    behavior_rank: str  # e.g., "Top", "Second", "Third"


def plot_combined_calibration_curve(
    data: Sequence[CalibrationData],
    filename: str = "combined_calibration_curve.pdf",
    x_axis_title: str = "Model Probability",
    y_axis_title: str = "Model Accuracy",
    chart_title: str = "",
    num_bins: int = 10,
    show_legend: bool = True,
) -> None:
    """
    Plots a combined calibration curve for multiple setups with distinct hues.

    Args:
        data (List[CalibrationData]): A list of CalibrationData instances.
        filename (str): The filename to save the plot. Defaults to "combined_calibration_curve.pdf".
        x_axis_title (str): Label for the x-axis. Defaults to "Model Probability".
        y_axis_title (str): Label for the y-axis. Defaults to "Model Accuracy".
        chart_title (str): The title of the chart.
        num_bins (int): Number of bins for calibration.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error

    # Convert data to a DataFrame for easier handling
    df = pd.DataFrame([data_item.dict() for data_item in data])

    # Initialize the plot with a larger size if necessary
    fig, ax = plt.subplots(figsize=(4, 4))

    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 15  # Reduced font size
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10

    # Get unique setups
    setups = df["setup_name"].unique()
    palette = ["#19c484", "#527fe8", "#d05881"]

    # Iterate over each setup and plot
    for idx, setup in reversed(list(enumerate(setups))):
        subset = df[df["setup_name"] == setup]
        if subset.empty:
            continue

        # Convert subset to list of tuples for binning
        subset_tuples = subset[["expected_prob", "predicted_prob"]].values.tolist()

        # Bin the data
        bin_means_x, bin_means_y = pandas_equal_frequency_binning(subset_tuples, num_bins=num_bins)

        # Calculate Mean Absolute Deviation (MAD)
        mad = mean_absolute_error([x * 100 for x in bin_means_x], [y * 100 for y in bin_means_y])

        # Plot the binned data
        ax.plot(
            bin_means_x,
            bin_means_y,
            marker="s",
            linestyle="-",
            color=palette[idx],
            label=f"{setup} (MAD={mad:.1f})",
        )

    # Add a y = x reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Set labels and title
    ax.set_xlabel(x_axis_title, fontsize=14)
    ax.set_ylabel(y_axis_title, fontsize=14)
    ax.set_title(chart_title)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Customize ticks to show percentages
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{int(x*100)}%" for x in np.arange(0, 1.1, 0.2)])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, 1.1, 0.2)])

    # Add legend
    if show_legend:
        # ax.legend(title="", loc="upper left")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title="", loc="upper left")

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Fine-tune as needed

    # Save the figure with no padding
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
