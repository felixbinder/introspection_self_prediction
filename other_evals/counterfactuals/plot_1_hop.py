# type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame from the data
data = {
    "Switched": ["Switch", "No switch", "Average", "Switch", "No switch", "Average"],
    "Model": [
        "Before finetune - gpt-3.5 on gpt-3.5",
        "Before finetune - gpt-3.5 on gpt-3.5",
        "Before finetune - gpt-3.5 on gpt-3.5",
        "After finetune on 1 hop - finetuned on finetuned",
        "After finetune on 1 hop - finetuned on finetuned",
        "After finetune on 1 hop - finetuned on finetuned",
    ],
    "Accuracy": [0.008, 0.99, 0.50, 0.407, 0.77, 0.589],
}

df = pd.DataFrame(data)

# Convert accuracy to percentage
df["Accuracy"] = df["Accuracy"] * 100

# Create a seaborn bar plot
# Modify the chart to include the actual numbers on the bars
plt.figure(figsize=(10, 6))
chart = sns.barplot(data=df, x="Switched", y="Accuracy", hue="Model")
chart.set_title("1 hop question peformance (+- 2% CI, ~2600 samples)")
chart.set_ylabel("Accuracy (%)")
chart.set_ylim(0, 105)  # Extend y-axis to show up to 100%
# remove the x-axis label
chart.set_xlabel("")

# Add the text annotations on each bar
for p in chart.patches:
    chart.annotate(
        f"{p.get_height():.0f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
    )

plt.legend(title="Model", loc="lower right")
plt.show()
