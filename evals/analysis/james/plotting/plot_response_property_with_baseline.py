import plotly.graph_objects as go

"""
mshifted predicting m: Average: 0.258741, SD: 0.438250, 95% CI: (0.226618, 0.290865), count: 715
Mode predicting m: 0.21958041958041957
mshifted predicting mshifted: Average: 0.416783, SD: 0.493371, 95% CI: (0.380619, 0.452947), count: 715
Mode predicting mshifted: 0.21678321678321677
"""
# Data


labels = ["Mft with P<br>predicting Mft", "Mft with P<br>predicting Mft with P"]
# accuracy = [24.0, 38.7]
# ci = [24.0 - 18.9, 38.7 - 32.9]
# modal_baseline = [0.2329749* 100, 0.21863799283154123 * 100]
accuracy = [25.8741, 41.6783]
ci = [25.8741 - 22.6618, 41.6783 - 38.0619]
modal_baseline = [21.958041958041957, 21.67832167832168]
colors = ["#636EFA", "#00CC96"]

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

# Add modal baseline with star markers
fig.add_trace(
    go.Scatter(
        x=labels,
        y=modal_baseline,
        mode="markers",
        name="Modal Baseline",
        marker=dict(symbol="star", size=8, color="black"),
    )
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
