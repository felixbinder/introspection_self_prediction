from typing import Sequence
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def plot_regression(tups: Sequence[tuple[float, bool]], x_axis_title: str = "Probability", y_axis_title: str = "Meta-level accuracy", chart_title: str ="") -> None:
    # Separate the probabilities and outcomes
    probabilities = [tup[0] for tup in tups]
    outcomes = [tup[1] for tup in tups]
    # Convert booleans to integers (0 or 1)
    outcomes = [int(outcome) for outcome in outcomes]

    # # Calculate the R coefficient
    # r, _ = stats.pearsonr(probabilities, outcomes)

    # Create the plot
    plt.figure(figsize=(10, 6))
    custom_bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Use seaborn's regplot with automatic binning
    sns.regplot(
        x=probabilities,
        y=outcomes,
        x_bins=custom_bins,
        scatter_kws={"alpha": 0.5},  # Add some transparency to points
        line_kws={"color": "red"},
    )
    # Set labels and title
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(chart_title)

    # Set x-axis limits
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.1)  # Set y-axis limits to show full range of binary outcome
    # Show the plot
    plt.grid(True)
    plt.show()

# Example usage
# example_data = [(0.1, False), (0.3, True), (0.5, True), (0.7, True), (0.9, True),
#                 (0.2, False), (0.4, True), (0.6, False), (0.8, True), (1.0, True)]
# plot_regression(example_data)