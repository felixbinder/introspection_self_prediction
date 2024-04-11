#!/usr/bin/env bash

set -ex

limit=1000
note="2024-04-11-classifier-3.5-turbo-limit-$limit"

python -m scripts.sweep_object_and_meta_levels \
        --study_name="jailbreaks" \
        --model_configs="gpt-3.5-turbo,gpt-4-turbo" \
        --task_configs="harmbench" \
        --response_property_configs="jailbroken" \
        --overrides="limit=$limit, strings_path=none, note=$note,
            cache_dir=/shared/exp/rajashree/jailbreaks/cache,
            language_model.logprobs=5,
            +response_property.language_model.logprops=5,
            +response_property.language_model.model=gpt-3.5-turbo,
            +response_property.exclusion_rule_groups=[]" \
        --meta_overrides="prompt=meta_level/bare"
