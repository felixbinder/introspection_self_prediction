import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["llama-70b", "gpt-3.5", "gpt-4"]
self_prediction = [65, 50, 63]
cross_prediction = [40, 45, 50]

# Calculate advantage
advantage_self = np.array(self_prediction) - np.array(cross_prediction)

# Positions and width for the bars
x = np.arange(len(models))
width = 0.6  # Width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Cross Prediction as the base
cross_bars = ax.bar(x, cross_prediction, width, label="Cross Prediction by GPT-4o", color="pink", edgecolor="black")

# Plot Advantage on top of Cross Prediction
advantage_bars = ax.bar(
    x,
    advantage_self,
    width,
    bottom=cross_prediction,
    label="Self-Prediction Advantage",
    color="blue",
    edgecolor="black",
    hatch="//",
    alpha=0.5,
)

# Add labels and title
ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Self Prediction vs. Cross Prediction with Advantage Shading", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# Add legend
ax.legend()


# Add data labels
def add_labels(bars, labels):
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_y() + height / 2.0,
                f"{label}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )


add_labels(cross_bars, labels=cross_prediction)
add_labels(advantage_bars, advantage_self)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()
