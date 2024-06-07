


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

python -m scripts.run_object_level

python -m evals.run_finetuning study_name={ft_study} language_model='projects/351298396653/locations/us-central1/endpoints/1362644551512096768' notes={notes}

python -m evals.run_object_level study_name=finetuning_demo3 language_model=projects/351298396653/locations/us-central1/endpoints/1362644551512096768 task=wikipedia task.set=train prompt=object_level/minimal limit=50
python -m evals.run_object_level --study_name=finetuning_demo3 --language_model='projects/351298396653/locations/us-central1/endpoints/1362644551512096768' --task='{"wikipedia": ["identity", "sentiment"]}' --task.set=train --prompt=object_level/minimal --limit=50

python -m evals.run_object_level study_name=finetuning_demo3 language_model=finetuned/finetuning_demo2/projects_351298396653_locations_us-central1_endpoints_1362644551512096768 task=wikipedia task.set=train prompt=object_level/minimal limit=50

python -m evals.run_object_level study_name=finetuning_demo3 language_model=gpt-3.5-turbo task=wikipedia task.set=train prompt=object_level/minimal limit=50

model_configs=""
projects/351298396653/locations/us-central1/endpoints/1362644551512096768


python -m scripts.sweep_full_study \
--study_name="finetuning_demo2" \
--model_configs="gpt-3.5-turbo" \
--val_only_model_configs="gpt-4" \
--tasks='{"wikipedia": ["identity", "sentiment"]}' \
--val_tasks='{"number_triplets": ["is_even"]}' \
--prompt_configs='minimal' \
--n_object_train=100 \
--n_object_val=25 \
--n_meta_val=50


python -m scripts.sweep_full_study \
--study_name="finetuning_demo_gemini" \
--model_configs="gemini-1.0-pro-002" \
--val_only_model_configs="gpt-3.5-turbo" \
--tasks='{"wikipedia": ["identity", "sentiment"]}' \
--val_tasks='{"number_triplets": ["is_even"]}' \
--prompt_configs='minimal' \
--n_object_train=100 \
--n_object_val=25 \
--n_meta_val=50



python -m scripts.sweep_full_study \
--study_name="training_on_everything_may_15_w_gemini" \
--model_configs="gpt-3.5-turbo,gemini-1.0-pro-002" \
--tasks='{"wikipedia": ["identity", "syllable_count", "first_character", "last_character"],  	"dear_abbie": ["identity", "sentiment", "dear_abbie/sympathetic_advice"], 	"number_triplets": ["identity", "is_even", "last_character"], 	"daily_dialog": ["identity", "syllable_count", "first_character", "last_character"],  	"personal_preferences": ["identity", "syllable_count", "first_character", "last_character"],  	"self_referential": ["identity", "syllable_count", "first_character", "last_character"],  	"writing_stories": ["identity", "sentiment", "first_character", "first_word", "writing_stories/good_ending", "writing_stories/main_character_name"]}' \
--prompt_configs='minimal'
