from omegaconf import OmegaConf

from evals.locations import CONF_DIR


def read_model_id_from_model_config(model_config: str) -> str:
    return OmegaConf.load(f"{CONF_DIR}/language_model/{model_config}.yaml").model
