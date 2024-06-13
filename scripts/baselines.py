# SUMMARY OF STEPS

# Run object level for both training and validation
# Run meta level for training and validation

# Create finetuning config for meta level and object level, need to pass dir of responses
# Create finetuning dataset for meta level and object level, pass self_training_baseline=True to search for meta config files

# full
# python -m scripts.sweep_full_study  \
# --study_name="baseline-test-full" \
# --model_configs="gpt-3.5-turbo" \
# --val_only_model_configs="gpt-4-turbo" \
# --tasks='{"number_triplets": ["identity"]}' \
# --val_tasks='{"wikipedia": ["identity"]}' \
# --prompt_configs='minimal' \
# --n_object_train=50 \
# --n_object_val=20 \
# --n_meta_train=50 \
# --n_meta_val=20


# run meta level sampling for some config


(
    "python -m evals.run_meta_level study_name=baseline-test"
    "language_model=gpt-3.5-turbo task=wikipedia response_property=identity"
    "task.set=train prompt=meta_level/minimal limit=20"
    "strings_path=/Users/milesturpin/Dev/nyu/introspection_self_prediction_astra/exp/smol_sweep_demo3/divergent_strings_wikipedia.csv"
)
# python -m evals.run_meta_level study_name=baseline-test language_model=gpt-3.5-turbo task=wikipedia response_property=identity task.set=train prompt=meta_level/minimal limit=20 strings_path=/Users/milesturpin/Dev/nyu/introspection_self_prediction_astra/exp/smol_sweep_demo3/divergent_strings_wikipedia.csv


# Path(EXP_DIR / self.args.study_name).mkdir(parents=True, exist_ok=True)

# python -m evals.run_object_level study_name=baseline-test language_model=gpt-3.5-turbo task=wikipedia task.set=train prompt=object_level/minimal limit=20
# Just run this on object level because for some stupid reason I need this to create meta level


# divergent strings is just a csv with a 'string' field
# divergent strings form the set of object level prompts that are then formatted
# So in this case since we want to run meta level responses on the whole training set,
# need to create a dummy divergent strings file for the training set
# yeah so no filtering needed on inputs for training,
# only filtering at eval time will be on the validation set to get inputs that diverge on the object level

# change limit to 2000 later


# change exp_dir in config_meta_level.yaml to add task set

# /Users/milesturpin/Dev/nyu/introspection_self_prediction_astra/exp/baseline-test/meta_level_gpt-3.5-turbo-1106_wikipedia_task_0_shot_True_seed_meta_level_minimal_prompt_identity_resp_train_task__note


# create dataset
# Going to need to

# create  finetuning dataset by indicating whether to use object level or meta level responses
# just need to indicate train val for meta object
# python -m evals.create_finetuning_dataset study_name=baseline-test dataset_folder=gpt-3.5-turbo finetune_models='gpt-3.5-turbo' self_training_baseline=True

# ok create_finetuning_dataset looks at all of the finetune dataset configs in
# exp/finetuning/baseline-test/gpt-3.5-turbo/gpt-3.5-turbo_wikipedia_identity_minimal.yaml
# so I need create the meta level configs and then glob the meta level ones


# Ok how do I go hard to the hoop here?
# - Just write this out a normal training set and just hardcode to set to true
# then just run!


# ok throwing error during create finetuning dataset because can't find identity property in one of the files. It's looking in the obejct level files
# Why doesn't obj level have identity property? When is this added usually?
#
# Are the meta level files getting the identity extracted?
