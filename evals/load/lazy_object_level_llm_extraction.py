import subprocess
from pathlib import Path

import pandas as pd


def lazy_add_response_property_to_object_level(object_df, object_cfg, response_property_name):
    if response_property_name not in object_df.columns:
        # we need to run the property extraction command
        run_property_extraction_command = f"python -m evals.run_property_extraction response_property={response_property_name} dir={object_cfg.exp_dir}"
        print(
            f"Response property {response_property_name} not in object level dataframe. Running property extraction with `{run_property_extraction_command}`."
        )
        # run the shell command
        main_path = Path(__file__).parent.parent.parent  # get the repo directory
        subprocess.run(run_property_extraction_command, shell=True, check=True, cwd=main_path)
        # now the file in the exp_dir should have the response property
        read_path = object_cfg.exp_dir + "/" + f"data{object_cfg.seed}.csv"
        updated_object_df = pd.read_csv(read_path, dtype=str)
        assert (
            response_property_name in updated_object_df.columns
        ), f"Response property {response_property_name} not in {read_path}."
        for column in updated_object_df.columns:
            if column not in ["complete"]:
                updated_object_df[column] = updated_object_df[column].astype(str)
        # load in the new column from the object level dataframe into the current one by joining on string
        object_df = pd.merge(object_df, updated_object_df[["string", response_property_name]], on="string")
        print(f"Loaded response property {response_property_name} from object level dataframe.")
    return object_df


def lazy_add_response_property_to_object_level_from_cfg_df_dict(cfg_df_dict, response_property_name):
    """The same as above, only on the entire dictionary of cfgs and dataframes."""
    for cfg, df in cfg_df_dict.items():
        df = lazy_add_response_property_to_object_level(df, cfg, response_property_name)
        cfg_df_dict[cfg] = df
    return cfg_df_dict
