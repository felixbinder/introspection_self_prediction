# type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame from the data

first_model = "gpt-3.5 on gpt-3.5"
second_model = "9GYUm36T on 9GYUm36T (Felix's trained on everything)"
data = {
    "Does the model actually get the question correct": [
        "Gets question correct",
        "Gets question wrong",
        "Average",
        "Gets question correct",
        "Gets question wrong",
        "Average",
    ],
    "Model": [first_model, first_model, first_model, second_model, second_model, second_model],
    "Accuracy": [0.884, 0.162, 0.524, 0.931, 0.138, 0.534],
}

df = pd.DataFrame(data)

# Convert accuracy to percentage
df["Accuracy"] = df["Accuracy"] * 100

# Create a seaborn bar plot
# Modify the chart to include the actual numbers on the bars
plt.figure(figsize=(10, 6))
chart = sns.barplot(data=df, x="Does the model actually get the question correct", y="Accuracy", hue="Model")
chart.set_title("Can models say whether they get a question correct?")
chart.set_ylabel("Accuracy (%)")
chart.set_ylim(0, 100)  # Extend y-axis to show up to 100%
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

plt.legend(title="Model", loc="upper right")
plt.show()
