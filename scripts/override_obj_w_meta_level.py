import csv
import os
import traceback

import numpy as np
import pandas as pd

from evals.analysis.analysis_helpers import (
    merge_object_and_meta_dfs_and_run_property_extraction,
)
from evals.analysis.loading_data import load_dfs_with_filter
from evals.locations import EXP_DIR
from evals.utils import get_maybe_nested_from_dict


def override_obj_level():
    #  relevant files: compliance_checks.py, loading_data.py, create_data_files_for_baseline_model.ipynd, object_vs_meta

    # given study name
    # how do I get the input object level df? in which step are the response properties written out all to the same notebook?

    # loop over object level dfs, get corresponding meta dfs, see
    # iterate over, do a join based on string , see if can reuse code
    # check for compliance, add option to also drop based on correctness

    # How do I use this modified object level df at the end?

    # for object_config, object_df in object_dfs.items():
    #     for meta_config, meta_df in meta_dfs.items():

    # load the dataframes with configs as keys

    STUDY_FOLDER = "may20_thrifty_sweep"

    CONDITIONS = {
        # see `analysis/loading_data.py` for details
        ("task", "set"): ["train"],
        # ("task", "name"): ["number_triplets"],
    }
    study_folder = EXP_DIR / (STUDY_FOLDER)
    unedited_study_folder = EXP_DIR / (STUDY_FOLDER + "_unedited")

    dfs = load_dfs_with_filter(unedited_study_folder, CONDITIONS, exclude_noncompliant=False)
    print(f"Loaded {len(dfs)} dataframes from {STUDY_FOLDER}")

    # os.makedirs(unedited_study_folder)
    # for cfg in dfs.keys():
    #     if not (unedited_study_folder / os.path.basename(cfg.exp_dir)).exists():
    #         os.system(f"cp -r {cfg.exp_dir} {unedited_study_folder / os.path.basename(cfg.exp_dir)}")

    # def restore_unedited():
    #     for cfg in dfs.keys():
    #         os.system(f"cp -r {unedited_study_folder / os.path.basename(cfg.exp_dir)} {cfg.exp_dir}")

    # import ipdb; ipdb.set_trace()

    print(f"Loaded {len(dfs)} dataframes in total")

    def is_base_config(config):
        return config["prompt"]["method"].startswith("object") or config["prompt"]["method"].startswith("base")

    object_dfs = {config: df for config, df in dfs.items() if is_base_config(config)}
    raw_object_dfs = {
        config: pd.read_csv(str(config.exp_dir).replace(str(study_folder), str(unedited_study_folder)) + "/data0.csv")
        for config, _ in object_dfs.items()
    }
    meta_dfs = {config: df for config, df in dfs.items() if not is_base_config(config)}
    print(f"Loaded {len(object_dfs)} base and {len(meta_dfs)} self-prediction dataframes")

    print("We have the following datasets:")
    datasets = set([get_maybe_nested_from_dict(k, ("task", "name")) for k in object_dfs.keys()])
    print(datasets)

    print("We have the following response properties:")
    response_properties = set([get_maybe_nested_from_dict(k, ("response_property", "name")) for k in meta_dfs.keys()])
    print(response_properties)

    def filter_by_dataset(dfs, dataset):
        return {
            config: df for config, df in dfs.items() if get_maybe_nested_from_dict(config, ("task", "name")) == dataset
        }

    def filter_by_dataset_and_response_property(dfs, dataset, response_property):
        return {
            config: df
            for config, df in dfs.items()
            if get_maybe_nested_from_dict(config, ("task", "name")) == dataset
            and get_maybe_nested_from_dict(config, ("response_property", "name")) == response_property
        }

    def filter_by_lm(dfs, lm):
        return {
            config: df
            for config, df in dfs.items()
            if get_maybe_nested_from_dict(config, ("language_model", "model")) == lm
        }

    # object_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in object_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(object_dfs.keys())}
    # meta_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in meta_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(meta_dfs.keys())}

    lens = []

    # we need to make groups of all models that belong together
    # object_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in object_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(object_dfs.keys())}
    # meta_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in meta_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(meta_dfs.keys())}

    all = {}
    # for object_group, _object_dfs in object_dfs_groups.items():
    #     for meta_group, _meta_dfs in meta_dfs_groups.items():
    #         all_joint_df = pd.DataFrame()
    #         for object_dfs in _object_dfs:
    #             for object_config, object_df in object_dfs.items():
    #                 for meta_dfs in _meta_dfs:

    for object_config, object_df in object_dfs.items():
        # all_joint_df = pd.DataFrame()
        all_joint_df = object_df.copy().set_index("string")
        for meta_config, meta_df in filter_by_lm(
            filter_by_dataset(meta_dfs, object_config["task"]["name"]), object_config["language_model"]["model"]
        ).items():
            # compute joint df

            # try:
            # joint_df = merge_object_and_meta_dfs(object_df, meta_df, meta_config.response_property.name)
            joint_df = merge_object_and_meta_dfs_and_run_property_extraction(
                object_df,
                meta_df,
                object_config,
                meta_config,
            )
            # import ipdb; ipdb.set_trace()
            joint_df = joint_df.set_index("string")
            pre_filter_len = len(joint_df)
            joint_df = joint_df[joint_df["compliance_meta"] == True]
            resp_prop_name = meta_config.response_property.name + "_meta"
            joint_df = joint_df[resp_prop_name]
            # all_joint_df = pd.concat([all_joint_df, joint_df])
            all_joint_df = all_joint_df.join(joint_df)

            lens.append((len(joint_df), pre_filter_len))

            all[object_config] = all_joint_df

    print(lens)

    mapping = {
        "string": "string",
        "sentiment": "sentiment_meta",
        "sentiment_complete": "sentiment_complete",
        "syllable_count_complete": "syllable_count_complete",
        "syllable_count": "syllable_count_meta",
        "writing_stories/main_character_name_complete": "writing_stories/main_character_name_complete",
        "writing_stories/main_character_name": "writing_stories/main_character_name_meta",
        "response": "response_meta",
        "first_word": "first_word_meta",
        # "complete": "complete_meta",
        "logprobs": "logprobs_meta",
        "first_character": "first_character_meta",
        "last_character": "last_character_meta",
        "identity": "identity_meta",
        "is_even": "is_even_meta",
        "raw_response": "raw_response_meta",
        "last_word_repeated": "last_word_repeated_meta",
        "last_char_repeated": "last_char_repeated_meta",
        "nonlast_word_repeated": "nonlast_word_repeated_meta",
        "any_word_repeated": "any_word_repeated_meta",
        "compliance": "compliance_meta",
        "first_logprobs": "first_logprobs_meta",
        "first_token": "first_token_meta",
    }
    reverse_mapping = {v: k for k, v in mapping.items()}

    def compute_compliance_rate_comparison(all_joint_df):
        object_comp = (all_joint_df["compliance_object"] == True).mean()
        meta_comp = (all_joint_df["compliance_meta"] == True).mean()
        # compliance_rate = all_joint_df.groupby("compliance_object")["compliance_meta"].mean()
        return compliance_rate

    def compute_similarity_rate(all_joint_df):
        similarity_rate = (all_joint_df["response_object"] == all_joint_df["response_meta"]).mean()
        return similarity_rate

    for cfg, joint_df in all.items():
        # if not (EXP_DIR / STUDY_FOLDER / 'old_object_level' / os.path.basename(cfg.exp_dir)).exists():
        #     os.system(f"cp -r {cfg.exp_dir} {EXP_DIR / STUDY_FOLDER / 'old_object_level' / os.path.basename(cfg.exp_dir)}")

        object_df_columns = raw_object_dfs[cfg].columns

        tmp = None
        try:
            # map actual object level columns to the fields in the joint df
            object_df_columns = [
                mapping[x]
                for x in object_df_columns
                if x not in ["complete", "logprobs", "response", "target", "metadata"]
            ]
            # ['is_even_meta', 'first_character_meta', 'last_character_meta']
            print("Dropping columns", set(object_df_columns) - set(joint_df.reset_index().columns))
            object_df_columns = list(set(object_df_columns) & set(joint_df.reset_index()))
            tmp = joint_df.reset_index()[object_df_columns].copy()

            # rename columns according to reverse mapping
            try:
                tmp.columns = [reverse_mapping[col] for col in tmp.columns]
            except Exception:
                # import ipdb

                # ipdb.set_trace()
                traceback.print_exc()
            tmp["logprobs"] = np.nan
            tmp["response"] = np.nan
            tmp["complete"] = True
            if "target" in raw_object_dfs[cfg].columns:
                tmp["target"] = raw_object_dfs[cfg]["target"]
            tmp["metadata"] = np.nan
            # save the new object level df
            tmp.to_csv(os.path.join(cfg.exp_dir, "data0.csv"), index=False, quotechar='"', quoting=csv.QUOTE_ALL)
        except Exception:
            # import ipdb

            # ipdb.set_trace()
            traceback.print_exc()


if __name__ == "__main__":
    override_obj_level()
