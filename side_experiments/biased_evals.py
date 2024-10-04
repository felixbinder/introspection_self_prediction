import pandas as pd
import plotly.express as px
import plotly.io as pio

"""
Saving BiasDetectAddAreYouSure to /Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectAddAreYouSure_results.csv
Saving BiasDetectWhatAnswerWithout to /Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectWhatAnswerWithout_results.csv
Saving BiasDetectAreYouAffected to /Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectAreYouAffected_results.csv
Saving KwikWillYouBeCorrect to /Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/KwikWillYouBeCorrect_results.csv
"""

# Step 1: Load and Combine CSV Files
csv_files = [
    "/Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectAddAreYouSure_results.csv",
    "/Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectAreYouAffected_results.csv",
    # '/Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/BiasDetectWhatAnswerWithout_results.csv',
    "/Users/jameschua/ml/introspection_self_prediction_astra/exp/other_evals/KwikWillYouBeCorrect_results.csv",
]
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Step 2: Prepare the Data
combined_df["meta_predicted_correctly"] = combined_df["meta_predicted_correctly"].astype(bool)

# Step 3: Mapping Dictionaries
object_model_mapping = {
    "gpt-4o-2024-05-13": "GPT-4o",
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo:resp-blin:A6imEZ8y": "Baseline Trained GPT-4o",
    "ft:gpt-4o-2024-05-13:dcevals-kokotajlo::A6Ji2P4o": "Self-Prediction Trained GPT-4o",
    # Add more mappings as needed
}

eval_name_mapping = {
    "BiasDetectAddAreYouSure": "Detecting<br>Are You Sure",
    "BiasDetectAreYouAffected": "Detecting<br>Anchor Bias",
    "KwikWillYouBeCorrect": "Detecting<br>Whether Correct",
    # "BiasDetectWhatAnswerWithout": "If
    # 'eval_name_id_1': 'Descriptive Evaluation Name 1',
    # 'eval_name_id_2': 'Descriptive Evaluation Name 2',
    # Add more mappings as needed
}

# Apply mappings to the DataFrame
combined_df["object_model_mapped"] = (
    combined_df["object_model"].map(object_model_mapping).fillna(combined_df["object_model"])
)
combined_df["eval_name_mapped"] = combined_df["eval_name"].map(eval_name_mapping).fillna(combined_df["eval_name"])

# Step 4: Calculate Accuracy and SEM
# Group by 'eval_name_mapped' and 'object_model_mapped' and calculate mean and SEM
grouped = combined_df.groupby(["eval_name_mapped", "object_model_mapped"])["meta_predicted_correctly"]

# Calculate mean accuracy
accuracy_df = grouped.mean().reset_index(name="Accuracy")

# Calculate SEM
sem_series = grouped.sem() * 100  # Multiply by 100 for percentage

# Assign SEM to accuracy_df
accuracy_df["SEM"] = sem_series.values

# Handle NaN SEM values (e.g., single observations)
accuracy_df["SEM"] = accuracy_df["SEM"].fillna(0)

# Convert accuracy to percentage
accuracy_df["Accuracy (%)"] = accuracy_df["Accuracy"] * 100

# Step 5: Define Desired Order for Models
desired_order = ["GPT-4o", "Baseline Trained GPT-4o", "Self-Prediction Trained GPT-4o"]

# Ensure 'object_model_mapped' is a categorical variable with the desired order
accuracy_df["object_model_mapped"] = pd.Categorical(
    accuracy_df["object_model_mapped"], categories=desired_order, ordered=True
)

# Step 6: Create the Bar Plot with Error Bars
fig = px.bar(
    accuracy_df,
    x="eval_name_mapped",
    y="Accuracy (%)",
    color="object_model_mapped",
    barmode="group",  # Ensure grouped bars
    text=accuracy_df["Accuracy (%)"].round(1),
    labels={
        "eval_name_mapped": "Evaluation Name",
        "Accuracy (%)": "Meta Predicted Correctly Accuracy (%)",
        "object_model_mapped": "Object Model",
    },
    category_orders={"object_model_mapped": desired_order},  # Correctly use 'category_orders'
    # title='Meta Predicted Correctly Accuracy by Evaluation Name and Object Model',
    error_y="SEM",  # Add error bars
    error_y_minus=accuracy_df["SEM"],  # Ensures symmetric error bars
)

# Update layout for better readability
fig.update_layout(xaxis_title="", yaxis_title="Accuracy (%)", legend_title="", template="plotly_white")

# Customize hover data to include SEM
fig.update_traces(
    hovertemplate=("<b>%{x}</b><br>" "Model: %{legendgroup}<br>" "Accuracy: %{y}%<br>" "SEM: %{error_y:.2f}%")
)

# Adjust plot dimensions and y-axis limits
fig.update_layout(height=150, width=750)  # Increased height for better readability
fig.update_yaxes(range=[30, 60])


pio.kaleido.scope.mathjax = None
# make sure there is no margin
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
# Save to PDF
fig.write_image("bias_detection.pdf")
fig.show()
