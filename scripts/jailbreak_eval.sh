#!/usr/bin/env bash

set -ex

python -m scripts.sweep_object_and_meta_levels \
        --study_name="jailbreaks" \
        --model_configs="gpt-3.5-turbo" \
        --task_configs="harmbench" \
        --response_property_configs="jailbroken" \
        --overrides="limit=1, strings_path=none,
            language_model.logprobs=5,
            +response_property.language_model.logprops=5,
            +response_property.language_model.model=gpt-4-turbo,
            +response_property.exclusion_rule_groups=[]" \
        --meta_overrides="prompt=meta_level/bare"
