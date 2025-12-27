from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CONFIG, BNB_CONFIG
from device_utils import detect_optimal_device

# Detect optimal device for tensor operations
DEVICE = detect_optimal_device()


def generate_response(model, tokenizer, prompt):
    alpaca_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(alpaca_prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.model_id)
tokenizer.pad_token = tokenizer.eos_token


# Test Prompt
test_instruction = "Write a Python function to add two numbers."

# --- BEFORE FINE-TUNE ---
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG.model_id, device_map="auto"
)
print("\n--- [BEFORE FINE-TUNE] ---")
print(generate_response(base_model, tokenizer, test_instruction))

# --- AFTER FINE-TUNE ---
# Load base model again (quantized to match training)
quant_model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG.model_id, quantization_config=BNB_CONFIG, device_map="auto"
)
ft_model = PeftModel.from_pretrained(quant_model, "./opt-125m-adapter")
print("\n--- [AFTER FINE-TUNE] ---")
print(generate_response(ft_model, tokenizer, test_instruction))
