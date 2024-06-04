from omegaconf import OmegaConf


def read_model_id_from_model_config(model_config: str) -> str:
    return OmegaConf.load(f"evals/conf/language_model/{model_config}.yaml").model
