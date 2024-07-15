import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("task_results.csv")


# Function to create and show the chart
def create_chart(df, title):
    # Create figure
    fig = go.Figure()

    # Add traces for before and after training
    for label in df["label"].unique():
        mask = df["label"] == label
        fig.add_trace(
            go.Bar(
                x=df[mask]["task"],
                y=df[mask]["accuracy"].str.rstrip("%").astype(float),
                name=label,
                text=df[mask]["accuracy"],
                textposition="auto",
                error_y=dict(type="data", array=df[mask]["error"].str.rstrip("%").astype(float), visible=True),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Task",
        yaxis_title="Accuracy (%)",
        barmode="group",
        yaxis=dict(range=[0, 100]),  # Set y-axis range from 0 to 100
        legend=dict(traceorder="normal"),
    )

    # Show the plot
    fig.show()


# Create the chart
create_chart(df, "Per Task: Model Accuracy Before and After Training with 95% CI")
