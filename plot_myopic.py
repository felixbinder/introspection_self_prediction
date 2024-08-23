import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('evidence_0.csv')

# reapply 1) as "Model before behavior change" and "Model after behavior change"
df["label"] = df["label"].apply(
    lambda x: "Model before behavior change" if "1)" in x else "Model after behavior change"
)

# Set the size of the plot
plt.figure(figsize=(10, 6))

# Create a barplot using seaborn with built-in error bars (standard deviation)
sns.barplot(x='label', y='object_response_property_answer', data=df, palette='Blues_d')

# Set the labels and title
plt.xlabel('Label')
plt.ylabel('% Choosing Myopic Reward')
plt.title('% Choosing Myopic Reward')

# Rotate the x labels for better readability
plt.xticks(ha='right')

# Show the plot
plt.tight_layout()
plt.show()
