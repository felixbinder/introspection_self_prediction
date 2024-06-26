
from pathlib import Path
from evals.apis.finetuning.run import FineTuneHyperParams, FineTuneParams, run_finetune
from evals.apis.finetuning.syncer import WandbSyncer
from evals.utils import load_secrets, setup_environment

left_out = """-val_tasks='{"writing_stories": ["first_word", "writing_stories/main_character_name"], "mmlu_cot": ["first_word"], "number_triplets": ["is_even"], "wealth_seeking": ["matches_wealth_seeking"], "power_seeking": ["matches_power_seeking"], "survival_instinct": ["matches_survival_instinct"], "myopic_reward": ["matches_myopic_reward"]}' --other_evals='["BiasDetectAreYouAffected", "KwikWillYouBeCorrect"]' --val_other_evals='["BiasDetectWhatAnswerWithout", "BiasDetectAddAreYouSure"]'"""

setup_environment()
params = FineTuneParams(
    model="gpt-4o-2024-05-13",
    hyperparameters=FineTuneHyperParams(
        n_epochs=1, learning_rate_multiplier=1, batch_size=16,
    ),
    suffix="james-gpt4o-sweep"
)


# try to find the data files
# defaults to cfg.study_dir / "train_dataset.jsonl"
data_path = Path("other_evals_combined_train_dataset.jsonl")

if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at {data_path}")

# load secrets
secrets = load_secrets("SECRETS")

syncer = WandbSyncer.create(project_name="jun25_leave_out_repsonse_prop_gpt4o".replace("/", "_"), notes=f"gpt-4o trained on gpt-3.5 with left out response properties {left_out}")

model_id = run_finetune(
    params=params,
    data_path=data_path,
    syncer=syncer,
    ask_to_validate_training=True,
    # organisation=org,
)