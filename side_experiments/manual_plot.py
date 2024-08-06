import plotly.graph_objects as go

# Data
labels = ["Mft with P<br>predicting Mft", "Mft with P<br>predicting Mft with P"]
accuracy = [24.0, 38.7]
ci = [24.0 - 18.9, 38.7 - 32.9]
colors = ["#636EFA", "#00CC96"]
modal_baseline = [50, 50]

# Create the bar chart
fig = go.Figure()

# Add existing bars
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

    # Add modal baseline
    go.Scatter(
        # x=x_positions,
        y=[modal_baseline[i]],
        mode="markers",
        name="Modal Baseline",
        marker=dict(symbol="star", size=8, color="black"),
        showlegend=True,
    )

# Update layout
fig.update_layout(
    title="Second character Meta-level Accuracy",
    xaxis_title="Predicting behavior with or without shifting prompt P",
    yaxis_title="Accuracy (%)",
    barmode="group",
    showlegend=True,
)

# Show the plot
fig.show()
