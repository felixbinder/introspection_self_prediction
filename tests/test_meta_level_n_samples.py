import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from evals import run_meta_level


def make_cfg(model: str, n_samples: int = 5):
    return OmegaConf.create(
        {
            "language_model": {"model": model},
            "n_samples": n_samples,
            "strings_path": "none",
            "base_dir": "some/base_dir",
            "base_seed": 0,
            "seed": 0,
            "n_shot": 1,
            "task": {},
            "response_property": {"name": "identity"},
        }
    )


class MetaLevelNSamplesTestCase(unittest.TestCase):
    def repeat_used_for(self, model: str) -> int:
        cfg = make_cfg(model)
        with patch.object(run_meta_level, "generate_few_shot_data") as gen:
            run_meta_level.setup_data_file(cfg, Path("exp"), Path("exp/raw_data0.csv"))
        return gen.call_args.kwargs["repeat"]

    def test_llama_data_file_is_not_repeated(self):
        # Models that are sampled once are never collated by mode-of-n, so the data
        # file must not contain repeated rows either.
        self.assertEqual(self.repeat_used_for("llama-70b-fireworks"), 1)

    def test_other_models_keep_configured_repeat(self):
        self.assertEqual(self.repeat_used_for("gpt-4o"), 5)

    def test_missing_n_samples_defaults_to_one(self):
        cfg = make_cfg("gpt-4o")
        del cfg["n_samples"]
        self.assertEqual(run_meta_level.get_n_samples(cfg), 1)


if __name__ == "__main__":
    unittest.main()
