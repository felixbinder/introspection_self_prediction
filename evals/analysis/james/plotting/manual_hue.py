import plotly.graph_objects as go

# Data
models = ["GPT3.5 (1106)", "GPT3.5 (1106)", "GPT-4", "GPT-4"]
training = [
    "Predicting behavior before training",
    "Predicting behavior after training",
    "Predicting behavior before training",
    "Predicting behavior after training",
]
accuracy = [23.5, 39.1, 25.9, 34.4]
ci_95 = [4.3, 4.9, 4.8, 5.2]

# Sort the data, but put GPT-4 last
sorted_data = sorted(zip(models, training, accuracy, ci_95), key=lambda x: (x[0] == "GPT-4", x[0], x[1]))
models, training, accuracy, ci_95 = zip(*sorted_data)

# Create figure
fig = go.Figure()

# Add traces for before and after training
for t in ["Predicting behavior before training", "Predicting behavior after training"]:
    mask = [tr == t for tr in training]
    fig.add_trace(
        go.Bar(
            x=[m for m, cond in zip(models, mask) if cond],
            y=[a for a, cond in zip(accuracy, mask) if cond],
            name=t,
            text=[f"{a}%" for a, cond in zip(accuracy, mask) if cond],
            textposition="auto",
            error_y=dict(type="data", array=[ci for ci, cond in zip(ci_95, mask) if cond], visible=True),
        )
    )

# Update layout
fig.update_layout(
    title="Model Accuracy Before and After Training with 95% CI",
    xaxis_title="Model",
    yaxis_title="Accuracy (%)",
    barmode="group",
    yaxis=dict(range=[0, 60]),  # Set y-axis range from 0 to 60
    legend=dict(traceorder="normal"),
)

# Show the plot
fig.show()
