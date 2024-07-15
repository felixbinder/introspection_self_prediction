import pandas as pd
import plotly.graph_objects as go

# Load the data


# Function to create and show the chart
def create_chart(df, title, _sorted_properties=None):
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

        # Accuracy bars
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=df_sorted["accuracy"] * 100,
                name=label,
                text=df_sorted["accuracy"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside",
                # plot with bootstrap_upper,bootstrap_lower
                error_y=dict(type="data", array=df_sorted["error"] * 100, visible=True),
                # error_y=dict(
                #     type="data",
                #     array=(df_sorted["accuracy"] - df_sorted["bootstrap_lower"]) * 100,
                #     arrayminus=(df_sorted["bootstrap_upper"] - df_sorted["accuracy"]) * 100,
                #     visible=True,
                # ),
                width=bar_width,
            )
        )

        # Mode baseline markers
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=df_sorted["mode_baseline"] * 100,
                mode="markers",
                name="Mode Baseline",
                marker=dict(symbol="x", size=8, color="red"),
            )
        )

    renamed = [prop.replace("zMicro Average", "Micro Average") for prop in sorted_properties]
    # rename "writing_stores/main_character_name" to "main_character_name"
    renamed = [prop.replace("writing_stories/main_character_name", "main_character_name") for prop in renamed]

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


# # Calculate macro average
# def calculate_macro_average(df):
#     macro_avg = df.groupby("label").agg({"accuracy": "mean", "error": "mean", "mode_baseline": "mean"}).reset_index()
#     macro_avg["response_property"] = "Macro Average"
#     return macro_avg


def main(csv_name: str, title: str = "Response Properties: Model Accuracy with Mode Baseline and 95% CI"):
    df = pd.read_csv(csv_name)
    # macro_avg = calculate_macro_average(df)
    # df = pd.concat([df, macro_avg])

    # Sort the response properties alphabetically, but keep 'Macro Average' at the end
    # sorted_properties = sorted(df["response_property"].unique())
    sorted_properties = [
        "first_character",
        "second_character",
        "third_character",
        "is_one_of_given_options",
        "matches behavior",
    ]
    # sorted_properties.remove("Macro Average")
    # sorted_properties.append("Macro Average")

    # Create the chart
    create_chart(df, title=title, _sorted_properties=sorted_properties)


if __name__ == "__main__":
    csv_name = "gpt_4o_iteration_2_heldout_rp.csv"
    # csv_name = "gpt_35_iteration_3_held_out.csv"
    main(csv_name, title="GPT-4o before and after finetuning")
# csv_name = "double_finetune.csv"
# csv_name = "response_property_results.csv"
# main(csv_name, title="Actual / training prediction")
