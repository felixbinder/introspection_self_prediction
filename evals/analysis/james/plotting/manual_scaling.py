import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create the data
data = {
    "Cross-prediction training samples": [1000, 5000, 10000, 20000, 30000],
    "cross predicting": [34.0, 34.2, 35.2, 36.2, 36.2],
}
self_prediction_acc = 49.6
"""
GPT-4o cross predicting	30.80%	34.70%	35.20%	34.50%	35.10%
Llama self prediction accuracy at 30000 samples	0.485	0.485	0.485	0.485	0.485
"""
# data = {
#     'Cross-prediction training samples': [1000, 5000, 10000, 20000, 30000],
#     'cross predicting': [30.80, 34.70, 35.20, 34.50, 35.10]
# }
# self_prediction_acc = 48.5


df = pd.DataFrame(data)

# Set the font to Helvetica
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 11

# Create the plot
plt.figure(figsize=(4, 4))
sns.set_style("whitegrid")

# Plot line for GPT-4o cross predicting with error bars, but without markers
plt.errorbar(
    df["Cross-prediction training samples"],
    df["cross predicting"],
    yerr=0.5,
    fmt="-",
    color="#527fe8",
    label="Cross-prediction",
    capsize=3,
    capthick=1,
    ecolor="#527fe8",
    elinewidth=1,
    linewidth=3,
)

# Add horizontal line for GPT-4 self-prediction accuracy
plt.axhline(y=self_prediction_acc, color="#19c484", linestyle="-", label="Self-prediction", linewidth=3)

# Customize the plot
plt.xlabel("Cross-prediction training samples", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.ylim(30, self_prediction_acc + 5)

# Remove gridlines
plt.grid(False)

# Add value labels for GPT-4o cross predicting
for x, y in zip(df["Cross-prediction training samples"], df["cross predicting"]):
    plt.text(x, y + 0.7, f"{y}%", ha="left", va="bottom", fontsize=8)  # Increased y-coordinate by 0.7

# Set y-ticks
y_ticks = range(30, int(self_prediction_acc) + 6, 5)
plt.yticks(y_ticks)

# Set x-ticks at every sample point
x_ticks = df["Cross-prediction training samples"]
plt.xticks(x_ticks)

# Format x-tick labels to show 'k' for thousands
plt.gca().set_xticklabels([f"{int(x/1000)}k" for x in x_ticks])

# Despine
sns.despine()

# Add annotation for self-prediction accuracy
plt.annotate(
    f"GPT-4o self-prediction accuracy = {self_prediction_acc}%",
    xy=(15000, self_prediction_acc),
    xytext=(5000, self_prediction_acc + 2),
    color="black",
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#19c484"),
)

# Show the plot
plt.tight_layout()

# Save to pdf
plt.savefig("cross_predict_gpt4_scale_with_error_bars.pdf")
# plt.savefig("cross_predict_llama_scale_with_error_bars.pdf")
# plt.show()
