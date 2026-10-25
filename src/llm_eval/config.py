from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config

def validate_config(config):
    required_sections = {
        "project",
        "data",
        "evaluation",
        "training",
        "reproducibility",
    }

    missing = required_sections - config.keys()

    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    fractions = [
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["test_fraction"],
    ]

    if any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("All data fractions must be between 0 and 1.")

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Data fractions must sum to 1.")

    training = config["training"]

    if training["learning_rate"] <= 0:
        raise ValueError("Learning rate must be positive.")

    if training["epochs"] < 1 or training["batch_size"] < 1:
        raise ValueError("Epochs and batch size must be positive.")

    if training["lora_rank"] < 1 or training["lora_alpha"] < 1:
        raise ValueError("LoRA rank and alpha must be positive.")

    print("Configuration validation passed.")

if __name__ == "__main__":
    config = load_config()

    print("Project:", config["project"]["name"])
    print("Random seed:", config["project"]["seed"])
    print("Training method:", config["training"]["method"])
    print("Base model:", config["training"]["base_model"])
