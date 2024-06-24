import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("response_property_results.csv")


# Calculate macro average
def calculate_macro_average(df):
    macro_avg = df.groupby("label").agg({"accuracy": "mean", "error": "mean"}).reset_index()
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
    # Create figure
    fig = go.Figure()

    # Add traces for before and after training
    for label in df["label"].unique():
        mask = df["label"] == label
        df_sorted = df[mask].set_index("response_property").loc[sorted_properties].reset_index()
        fig.add_trace(
            go.Bar(
                x=df_sorted["response_property"],
                y=df_sorted["accuracy"] * 100,  # Convert to percentage
                name=name,
                text=df_sorted["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside",  # Place text outside the bar
                error_y=dict(type="data", array=df_sorted["error"] * 100, visible=True),  # Convert to percentage
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Response Property",
        yaxis_title="Accuracy (%)",
        barmode="group",
        yaxis=dict(range=[0, 100]),  # Set y-axis range from 0 to 100
        legend=dict(traceorder="normal"),
        xaxis=dict(tickangle=45, categoryorder="array", categoryarray=sorted_properties),
    )

    # Show the plot
    fig.show()


# Create the chart
create_chart(df, "Response Properties: Model Accuracy Before and After Training with 95% CI", sorted_properties)
