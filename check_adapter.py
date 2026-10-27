from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = Path("qwen-routing-lora")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER_PATH),
    local_files_only=True,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
)

model.eval()

print("\nAdapter loaded successfully.")
print("Device:", next(model.parameters()).device)
print("Tokenizer vocabulary size:", len(tokenizer))

messages = [
    {
        "role": "user",
        "content": (
            "I noticed a payment on my account that I don't recognize. "
            "Please classify this message. Return JSON with "
            "'route' and 'escalate' fields."
        ),
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\nGenerated response:")
print(response)