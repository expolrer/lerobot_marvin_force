from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train import apply_smoke_overrides  # noqa: E402


def test_smoke_overrides_are_isolated_and_resumable() -> None:
    config = {
        "log_dir": "/ssd/force/logs/manifeel_usb/fcact",
        "runs": [
            {
                "output_dir": "/ssd/force/outputs/lerobot_marvin_force/manifeel_usb/fcact",
                "batch_size": 64,
                "num_workers": 8,
                "steps": 300_000,
                "save_freq": 1_000,
                "eval_steps": 5_000,
                "log_freq": 10,
                "push_to_hub": True,
                "wandb_enable": True,
            }
        ],
    }

    apply_smoke_overrides(config, 3)

    run = config["runs"][0]
    assert Path(config["log_dir"]).name == "smoke"
    assert Path(run["output_dir"]).parts[-2:] == ("smoke", "fcact")
    assert run["steps"] == 3
    assert run["save_freq"] == 1
    assert run["batch_size"] == 1
    assert run["num_workers"] == 0
    assert run["eval_steps"] == 0
    assert not run["push_to_hub"]
    assert not run["wandb_enable"]


def test_smoke_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_smoke_overrides({"log_dir": "/tmp/log", "runs": []}, 0)
