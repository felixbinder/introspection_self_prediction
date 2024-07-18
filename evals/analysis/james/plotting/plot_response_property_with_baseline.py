import pandas as pd
import plotly.graph_objects as go


# Function to create and show the chart
def create_chart(df, title, first_chart_color: str = "#636EFA", _sorted_properties=None):
    if _sorted_properties is None:
        sorted_properties = sorted(df["response_property"].unique())
    else:
        sorted_properties = _sorted_properties

    fig = go.Figure()

    # Calculate bar positions
    n_properties = len(sorted_properties)
    sorted_labels = sorted(df["label"].unique())
    n_labels = len(sorted_labels)
    bar_width = 0.8 / n_labels

    # Create color list
    colors = [first_chart_color, "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    for i, label in enumerate(sorted_labels):
        mask = df["label"] == label
        df_label = df[mask].set_index("response_property")

        # Filter sorted_properties to only include those present in df_label
        available_properties = [prop for prop in sorted_properties if prop in df_label.index]

        df_sorted = df_label.loc[available_properties].reset_index()

        # Calculate x-positions for bars and scatter points
        x_positions = [
            sorted_properties.index(prop) + (i - (n_labels - 1) / 2) * bar_width for prop in available_properties
        ]
        color_used = colors[i]
        print(f"color_used: {color_used} for label: {label}")

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
                marker_color=color_used,
            )
        )

        # Mode baseline markers
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=df_sorted["mode_baseline"] * 100,
                mode="markers",
                name="Mode Baseline",
                marker=dict(symbol="star", size=8, color="black"),
            )
        )

    renamed = [prop.replace("zMicro Average", "Micro Average") for prop in sorted_properties]
    renamed = [
        prop.replace("writing_stories/main_character_name", "main_character_name").replace("_", " ") for prop in renamed
    ]

    fig.update_layout(
        title=title,
        xaxis_title="Response Property",
        yaxis_title="Accuracy",
        barmode="group",
        yaxis=dict(range=[0, 100]),
        legend=dict(traceorder="normal"),
        xaxis=dict(tickmode="array", tickvals=list(range(n_properties)), ticktext=renamed),
    )
    # save as png 1080p
    fig.write_image("response_property_results.png", width=1920, height=1080)
    fig.show()


def main(csv_name: str, title: str = "Response Properties: Model Accuracy with Mode Baseline and 95% CI"):
    df = pd.read_csv(csv_name)

    # Create the chart
    # #636EFA pale blue
    #
    create_chart(df, title=title, _sorted_properties=None, first_chart_color="palevioletred")


if __name__ == "__main__":
    csv_name = "response_property_results.csv"
    main(csv_name, title="GPT-4o before and after finetuning")
