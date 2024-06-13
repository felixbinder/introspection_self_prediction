from evals.analysis.analysis_helpers import merge_object_and_meta_dfs
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

    STUDY_FOLDERS = [  # 🔵 within exp/
        # "baseline-test-full6"
        "may20_thrifty_sweep"
    ]

    CONDITIONS = {
        # see `analysis/loading_data.py` for details
        ("task", "set"): ["train"],
        # ("task", "name"): ["number_triplets"],
    }
    dfs = {}
    for STUDY_FOLDER in STUDY_FOLDERS:
        _dfs = load_dfs_with_filter(EXP_DIR / STUDY_FOLDER, CONDITIONS, exclude_noncompliant=False)
        dfs.update(_dfs)
        print(f"Loaded {len(_dfs)} dataframes from {STUDY_FOLDER}")
    print(f"Loaded {len(dfs)} dataframes in total")

    def is_base_config(config):
        return config["prompt"]["method"].startswith("object") or config["prompt"]["method"].startswith("base")

    object_dfs = {config: df for config, df in dfs.items() if is_base_config(config)}
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

    # object_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in object_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(object_dfs.keys())}
    # meta_dfs_groups = {cfg['language_model']['model']:[{k:v} for k,v in meta_dfs.items() if k['language_model']['model'] == cfg['language_model']['model']] for cfg in set(meta_dfs.keys())}

    lens = []
    for object_config, object_df in object_dfs.items():
        for meta_config, meta_df in filter_by_dataset(meta_dfs, object_config["task"]["name"]).items():
            # compute joint df

            # try:
            joint_df = merge_object_and_meta_dfs(object_df, meta_df, meta_config.response_property.name)
            # joint_df = merge_object_and_meta_dfs_and_run_property_extraction(
            #     object_df,
            #     meta_df,
            #     object_config,
            #     meta_config,
            # )
            lens.append(len(joint_df))
            if len(joint_df) == 9:
                # import ipdb

                # ipdb.set_trace()
                pass
            # except:
            #     # import ipdb

            #     # ipdb.set_trace()
            #     pass

            # can maybe just use merge_object_and_meta_dfs

            # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    override_obj_level()
