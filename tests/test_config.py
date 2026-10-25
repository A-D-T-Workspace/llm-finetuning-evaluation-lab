import copy

import pytest

from src.llm_eval.config import load_config, validate_config

def test_project_configuration_loads():
    config = load_config()

    assert config["project"]["seed"] == 42
    assert config["training"]["method"] == "lora"

def test_valid_configuration_passes():
    config = load_config()

    validate_config(config)

def test_invalid_data_fractions_fail():
    config = load_config()
    config = copy.deepcopy(config)
    config["data"]["train_fraction"] = 0.80

    with pytest.raises(ValueError, match="sum to 1"):
        validate_config(config)

def test_non_positive_learning_rate_fails():
    config = load_config()
    config = copy.deepcopy(config)
    config["training"]["learning_rate"] = 0

    with pytest.raises(ValueError, match="Learning rate"):
        validate_config(config)
