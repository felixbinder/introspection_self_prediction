# open in pandas exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv
import pandas as pd

from evals.analysis.james.object_meta import FlatObjectMeta
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

#
# before_finetune = read_jsonl_file_into_basemodel("gpt-3.5-turbo-0125_first_character_filtered.jsonl", FlatObjectMeta)
before_finetune = read_jsonl_file_into_basemodel("entropy_new_before_finetune.jsonl", FlatObjectMeta)

# ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl
after_finetune = read_jsonl_file_into_basemodel("entropy_new_after_finetune.jsonl", FlatObjectMeta)

before_finetune_object_response_property_answer: list[str] = before_finetune.map(
    lambda x: x.object_response_property_answer
)
after_finetune_object_response_property_answer: list[str] = after_finetune.map(
    lambda x: x.object_response_property_answer
)

import pandas as pd
import plotly.graph_objects as go

# Convert lists to pandas Series
before_series = pd.Series(before_finetune_object_response_property_answer, name="GPTo's distribution")
after_series = pd.Series(after_finetune_object_response_property_answer, name="GPTo_fton_GPTo distribution")

# Combine the series into a DataFrame
df = pd.concat([before_series, after_series], axis=1)

# Get value counts and calculate percentages
df_percentages = df.apply(lambda x: x.value_counts(normalize=True) * 100).reset_index()
df_percentages.columns = ["Response", "GPTo's distribution", "GPTo_fton_GPTo distribution"]

# Melt the DataFrame for easier plotting
df_melted = df_percentages.melt(id_vars="Response", var_name="Group", value_name="Percentage")

# Create the figure
fig = go.Figure()

# Add traces for each group
for group in ["GPTo's distribution", "GPTo_fton_GPTo distribution"]:
    df_group = df_melted[df_melted["Group"] == group]
    # sort by the response
    df_group = df_group.sort_values(by="Response")
    fig.add_trace(
        go.Bar(
            x=df_group["Response"],
            y=df_group["Percentage"],
            name=group,
            text=df_group["Percentage"].round(2).astype(str) + "%",
            textposition="auto",
            # color by group, "636efa" and "rgb(0 204 150)"
            marker=dict(color="rgb(99,110,250)" if group == "GPTo's distribution" else "rgb(0,204,150)"),
        )
    )

# Update layout
fig.update_layout(
    title="Distribution of Object Response Property Answers (Percentage)",
    xaxis_title="Response",
    yaxis_title="Percentage",
    barmode="group",
    bargap=0.15,
    bargroupgap=0.1,
    yaxis=dict(tickformat=".1f", ticksuffix="%"),
)

# Show the plot
fig.show()
