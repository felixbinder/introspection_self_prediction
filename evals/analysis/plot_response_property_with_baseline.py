import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("double_finetune.csv")

# Calculate macro average
def calculate_macro_average(df):
    macro_avg = df.groupby("label").agg({"accuracy": "mean", "error": "mean", "mode_baseline": "mean"}).reset_index()
    macro_avg["response_property"] = "Macro Average"
    return macro_avg

macro_avg = calculate_macro_average(df)
df = pd.concat([df, macro_avg])

# Sort the response properties alphabetically, but keep 'Macro Average' at the end
sorted_properties = sorted(df["response_property"].unique())
sorted_properties.remove("Macro Average")
sorted_properties.append("Macro Average")

# Function to create and show the chart
def create_chart(df, title, sorted_properties):
    fig = go.Figure()

    # Calculate bar positions
    n_properties = len(sorted_properties)
    n_labels = len(df["label"].unique())
    bar_width = 0.8 / n_labels

    for i, label in enumerate(df["label"].unique()):
        mask = df["label"] == label
        df_label = df[mask].set_index("response_property")
        
        # Filter sorted_properties to only include those present in df_label
        available_properties = [prop for prop in sorted_properties if prop in df_label.index]
        
        df_sorted = df_label.loc[available_properties].reset_index()

        # Calculate x-positions for bars and scatter points
        x_positions = [sorted_properties.index(prop) + (i - (n_labels - 1) / 2) * bar_width for prop in available_properties]

        # Accuracy bars
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=df_sorted["accuracy"] * 100,
                name=label,
                text=df_sorted["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside",
                error_y=dict(type="data", array=df_sorted["error"] * 100, visible=True),
                width=bar_width,
            )
        )

        # Mode baseline markers
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=df_sorted["mode_baseline"] * 100,
                mode="markers",
                name=f"{label} (Mode Baseline)",
                marker=dict(symbol="star", size=8, color="red"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Response Property",
        yaxis_title="Percentage",
        barmode="group",
        yaxis=dict(range=[0, 100]),
        legend=dict(traceorder="normal"),
        xaxis=dict(tickangle=45, tickmode="array", tickvals=list(range(n_properties)), ticktext=sorted_properties),
    )
    fig.show()

# Create the chart
create_chart(df, "Response Properties: Model Accuracy with Mode Baseline and 95% CI", sorted_properties)