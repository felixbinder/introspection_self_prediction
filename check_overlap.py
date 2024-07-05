# open in pandas exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv
import pandas as pd

#
object_df = pd.read_csv(
    "exp/jun20_training_on_everything/object_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_object_level_minimal_prompt_number_triplets_val_task__note/data0.csv"
)
print(f"Object level data shape: {object_df.shape}")

meta_df = pd.read_csv(
    "exp/jun20_training_on_everything/meta_level_ft_gpt-3.5-turbo-0125_dcevals-kokotajlo__9da15ENS_number_triplets_task_0_shot_True_seed_meta_level_minimal_prompt_first_character_resp__note/data0.csv"
)
print(f"Meta level data shape: {meta_df.shape}")
object_df_strings = object_df["string"]
meta_df_strings = meta_df["string"]
# calc overlap
overlap = object_df_strings.isin(meta_df_strings)
percentage = overlap.sum() / len(object_df_strings)
print(f"Overlap between object and meta level: {percentage:.2f} ({overlap.sum()} / {len(object_df_strings)})")
