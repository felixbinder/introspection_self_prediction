


<!-- python -m scripts.sweep_full_study \
--study_name="full_sweep_demo" \
--model_configs="gpt-3.5-turbo" \
--val_only_model_configs="gpt-4" \
--tasks='{"wikipedia": ["identity", "sentiment"], "dear_abbie": ["identity", "sentiment", "dear_abbie/sympathetic_advice"]}' \
--val_tasks='{"number_triplets": ["identity", "is_even"], "english_words": ["identity", "first_character"]}' \
--prompt_configs='minimal' \
--n_object_train=1000 \
--n_object_val=250 \
--n_meta_val=50 \
--skip_finetuning -->


python -m scripts.sweep_full_study \
--study_name="full_sweep_demo" \
--model_configs="gpt-3.5-turbo" \
--val_only_model_configs="gpt-4" \
--tasks='{"wikipedia": ["identity", "sentiment", "counterfactual_identity"]}' \
--val_tasks='{"number_triplets": ["identity", "is_even"], "english_words": ["identity", "first_character"]}' \
--prompt_configs='minimal' \
--n_object_train=1000 \
--n_object_val=250 \
--n_meta_val=50 \
--skip_finetuning
