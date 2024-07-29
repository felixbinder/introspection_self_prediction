import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Create a DataFrame from the data
data = {
    "model": ["GPT-3.5", "GPT-3.5", "GPT-4", "GPT-4"],
    "condition": ["Before", "After", "Before", "After"],
    "accuracy": [0.17626565051714752, 0.5509254218835057, 0.36136079189896464, 0.5985891455228126],
    "bootstrap_lower": [0.17261431682090364, 0.5456702504082744, 0.35146205484127885, 0.5888013425873251],
    "bootstrap_upper": [0.18032117583015786, 0.5559084104518236, 0.3719450449425418, 0.60860450563204],
    "mode_baseline": [0.36456178551986934, 0.400870985302123, 0.3657981567868927, 0.43269996586642395],
}

df = pd.DataFrame(data)

# Create the plot
fig = go.Figure()

# Define colors for before and after conditions
colors = {"Before": "palevioletred", "After": "#00CC96"}

# Calculate bar positions
n_conditions = len(df["condition"].unique())
n_models = len(df["model"].unique())
bar_width = 0.8 / n_conditions

for i, condition in enumerate(["Before", "After"]):
    df_condition = df[df["condition"] == condition]

    # Calculate x-positions for bars
    x_positions = [j + (i - (n_conditions - 1) / 2) * bar_width for j in range(n_models)]

    # Add modal baselines as stars
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=df_condition["mode_baseline"],
            mode="markers",
            name="Baseline (predicting modal answer)",
            marker=dict(symbol="star", size=8, color="black"),
            showlegend=False if i == 1 else True,
        )
    )

    # Add bars
    fig.add_trace(
        go.Bar(
            x=x_positions,
            y=df_condition["accuracy"],
            name=f"{condition} self-prediction training",
            marker_color=colors[condition],
            error_y=dict(
                type="data",
                symmetric=False,
                array=df_condition["bootstrap_upper"] - df_condition["accuracy"],
                arrayminus=df_condition["accuracy"] - df_condition["bootstrap_lower"],
            ),
            width=bar_width,
            text=df_condition["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
            textposition="outside",
        )
    )


# Update layout
fig.update_layout(
    title="",
    xaxis_title="Model",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 0.8], tickformat=".0%"),  # Set y-axis range from 0 to 1 and format as percentage
    legend_title="Legend",
    showlegend=True,
    # make legend text bigger
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0, title=None, font=dict(size=14)),
    barmode="group",  # Group bars for each model side by side
    xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["GPT-3.5", "GPT-4o"]),
)
fig.update_layout(height=500, width=400)


pio.kaleido.scope.mathjax = None
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image("summary_plot.pdf")

# Show the plot
# fig.show()
