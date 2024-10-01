import pandas as pd
import plotly.graph_objects as go
from git import Sequence


def wrap_label(label):
    # Make the first word the first line. Everything else the second line
    words = label.split(" ")
    return f"{words[0]}<br>{' '.join(words[1:])}"


def wrap_labels(labels):
    return [wrap_label(label) for label in labels]


# Function to create and show the chart
def create_chart(
    df,
    title,
    first_chart_color: str = "#636EFA",
    _sorted_properties: Sequence[str] = [],
    fix_ratio: bool = True,
    sorted_labels: Sequence[str] = [],
    pdf_name: str = "response_property_results.pdf",
    show_legend: bool = True,
):
    if len(_sorted_properties) == 0:
        sorted_properties = sorted(df["response_property"].unique())
    else:
        sorted_properties = _sorted_properties

    fig = go.Figure()

    # Calculate bar positions
    n_properties = len(sorted_properties)
    if len(sorted_labels) == 0:
        sorted_labels = sorted(df["label"].unique())
    n_labels = len(sorted_labels)
    bar_width = 0.8 / n_labels

    # Create color list
    colors = [
        first_chart_color,
        "#1ab87b",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

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

        # Mode baseline markers
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=df_sorted["mode_baseline"] * 100,
                mode="markers",
                name="Modal Baseline",
                marker=dict(symbol="star", size=8, color="black"),
                showlegend=True if i == 0 else False,
            )
        )

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

    renamed = [prop.replace("zMicro Average", "Average of properties") for prop in sorted_properties]
    # Upper case first letter
    renamed = [prop[0].upper() + prop[1:] for prop in renamed]

    fig.update_layout(
        title=title,
        yaxis_title="Accuracy",
        barmode="group",
        yaxis=dict(
            range=[0, 62],
            ticksuffix="%",  # Added to append '%' to each tick
            title_font=dict(size=18),
            tickfont=dict(size=16),  # Optional: Adjust tick label size
            showline=True,  # Optional: Show line
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.99,
            title=None,
            font=dict(size=16),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_properties)),
            ticktext=renamed,
            tickangle=0,
            tickfont=dict(size=18),
            showline=True,
        ),
        plot_bgcolor="white",
        # xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=False),
        # yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=False),
        font=dict(  # Added font settings
            family="Helvetica",
            size=14,  # Adjust the size as needed
            color="black",  # Adjust the color as needed
        ),
        margin=dict(l=0, r=50, t=2.0, b=40),  # Moved margin settings here
        height=330 if fix_ratio else None,
        width=400 if fix_ratio else None,
    )

    # Remove legend if not needed
    if not show_legend:
        fig.update_layout(showlegend=False)

    # Save as PDF
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig.write_image(pdf_name)

    fig.show()


def alt_main(csv_name: str, title: str = "Response Properties: Model Accuracy with Mode Baseline and 95% CI"):
    df = pd.read_csv(csv_name)

    create_chart(
        df,
        title=title,
        _sorted_properties=["GPT-4o", "Llama 70b", "GPT-3.5"],
        first_chart_color="palevioletred",
        show_legend=False,
        pdf_name="all_evidence_0.pdf",
    )


if __name__ == "__main__":
    csv_name = "evals/analysis/james/plotting/results_csvs/all_evidence_0.csv"
    alt_main(csv_name, title="")
