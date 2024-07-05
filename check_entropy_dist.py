# open in pandas exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv
import pandas as pd

from evals.analysis.james.object_meta import FlatObjectMeta
from other_evals.counterfactuals.api_utils import read_jsonl_file_into_basemodel

#
before_finetune = read_jsonl_file_into_basemodel("gpt-3.5-turbo-0125_first_character.jsonl", FlatObjectMeta)

# ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl
after_finetune = read_jsonl_file_into_basemodel(
    "ft:gpt-3.5-turbo-0125:dcevals-kokotajlo::9da15ENS_first_character.jsonl", FlatObjectMeta
)

before_finetune_object_response_property_answer: list[str] = before_finetune.map(
    lambda x: x.object_response_property_answer
)
after_finetune_object_response_property_answer: list[str] = after_finetune.map(
    lambda x: x.object_response_property_answer
)

import pandas as pd
import plotly.graph_objects as go

# Convert lists to pandas Series
before_series = pd.Series(before_finetune_object_response_property_answer, name="Before Fine-tuning")
after_series = pd.Series(after_finetune_object_response_property_answer, name="After Fine-tuning")

# Combine the series into a DataFrame
df = pd.concat([before_series, after_series], axis=1)

# Get value counts and calculate percentages
df_percentages = df.apply(lambda x: x.value_counts(normalize=True) * 100).reset_index()
df_percentages.columns = ["Response", "Before Fine-tuning", "After Fine-tuning"]

# Melt the DataFrame for easier plotting
df_melted = df_percentages.melt(id_vars="Response", var_name="Group", value_name="Percentage")

# Create the figure
fig = go.Figure()

# Add traces for each group
for group in ["Before Fine-tuning", "After Fine-tuning"]:
    df_group = df_melted[df_melted["Group"] == group]
    fig.add_trace(
        go.Bar(
            x=df_group["Response"],
            y=df_group["Percentage"],
            name=group,
            text=df_group["Percentage"].round(2).astype(str) + "%",
            textposition="auto",
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
