import pandas as pd
import plotly.graph_objects as go

from evals.analysis.james.object_meta import ObjectAndMeta
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

# Read the data
before_finetune = read_jsonl_file_into_basemodel("gpt-4o-2024-05-13_first_character.jsonl", ObjectAndMeta)
after_finetune = read_jsonl_file_into_basemodel(
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::9danhPzM_first_character.jsonl", ObjectAndMeta
)

# Extract the object_response_property_answer
before_finetune_object_response_property_answer: list[str] = before_finetune.map(
    lambda x: x.object_response_property_answer
)
after_finetune_object_response_property_answer: list[str] = after_finetune.map(
    lambda x: x.object_response_property_answer
)

# Convert lists to pandas Series
before_series = pd.Series(before_finetune_object_response_property_answer, name="GPTo's distribution")
after_series = pd.Series(after_finetune_object_response_property_answer, name="GPTo_fton_GPTo distribution")

# Combine the series into a DataFrame
df = pd.concat([before_series, after_series], axis=1)

# Get value counts
df_counts = df.apply(lambda x: x.value_counts()).reset_index()
df_counts.columns = ["Response", "GPTo's distribution", "GPTo_fton_GPTo distribution"]

# Melt the DataFrame for easier plotting
df_melted = df_counts.melt(id_vars="Response", var_name="Group", value_name="Count")

# Create the figure
fig = go.Figure()

# Add traces for each group
for group in ["GPTo's distribution", "GPTo_fton_GPTo distribution"]:
    df_group = df_melted[df_melted["Group"] == group]
    fig.add_trace(
        go.Bar(
            x=df_group["Response"],
            y=df_group["Count"],
            name=group,
            text=df_group["Count"],
            textposition="auto",
            marker=dict(color="rgb(99,110,250)" if group == "GPTo's distribution" else "rgb(0,204,150)"),
        )
    )

# Update layout
fig.update_layout(
    title="Distribution of Object Response Property Answers (Count)",
    xaxis_title="Response",
    yaxis_title="Count",
    barmode="group",
    bargap=0.15,
    bargroupgap=0.1,
)

# Show the plot
fig.show()
