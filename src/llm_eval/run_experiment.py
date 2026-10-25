from src.llm_eval.config import load_config
from src.llm_eval.reproducibility import set_seed

def main():
    config = load_config()

    seed = config["project"]["seed"]
    deterministic = config["reproducibility"]["deterministic"]

    set_seed(seed, deterministic=deterministic)

    print("Experiment initialized successfully.")
    print("Random seed:", seed)
    print("Deterministic mode:", deterministic)

if __name__ == "__main__":
    main()
