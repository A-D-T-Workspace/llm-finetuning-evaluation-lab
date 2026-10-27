def main():
    print(
        "This evaluation entry point is deprecated: it generated "
        "placeholder predictions, not real model outputs."
    )
    print(
        "For the actual base-model vs. LoRA evaluation, run:"
    )
    print("python -m src.llm_eval.compare_models")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
