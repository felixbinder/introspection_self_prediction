import plotly.graph_objects as go

# Data
labels = ["gpt35 predicting gpt35", "gpt4o_fton_gpt35 predicting gpt 35", "gpt35_fton_gpt35 predicting gpt 35"]
accuracy = [26, 40.9, 46.9]
ci = [1.25, 2.9, 2.6]
colors = ["green", "blue", "red"]

# Create the bar chart
fig = go.Figure()

for i in range(len(labels)):
    fig.add_trace(
        go.Bar(
            x=[labels[i]],
            y=[accuracy[i]],
            error_y=dict(type="data", array=[ci[i]], visible=True),
            marker_color=colors[i],
            name=labels[i],
        )
    )

# Update layout
fig.update_layout(
    title="Accuracy in Predicting GPT-3.5",
    xaxis_title="Model",
    yaxis_title="Accuracy (%)",
    barmode="group",
    showlegend=False,
)

# Show the plot
fig.show()
