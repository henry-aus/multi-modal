import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from config import MODEL_CONFIG, BNB_CONFIG
from device_utils import detect_optimal_device
import os

# Set environment variables for accelerate
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"  # Disable mixed precision to avoid device conflicts
os.environ["ACCELERATE_CPU_ONLY"] = "false" if detect_optimal_device() != "cpu" else "true"

# Detect optimal device for training
DEVICE = detect_optimal_device()
# Force CPU if having device issues - uncomment the line below
# DEVICE = "cpu"
print(f"Using device: {DEVICE}")


# Load Dataset
dataset = load_dataset(MODEL_CONFIG.dataset_id, split="train")


# Formatting Function for Alpaca
def formatting_prompts_func(examples):
    """Format examples for Alpaca-style instruction following."""
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}


# Apply formatting to dataset
dataset = dataset.map(
    formatting_prompts_func, batched=True, remove_columns=dataset.column_names
)

# Load Model & Tokenizer
# Note: When using quantization, the model needs to stay on CPU initially
# Let accelerate handle device placement during training preparation
model = AutoModelForCausalLM.from_pretrained(
    MODEL_CONFIG.model_id,
    quantization_config=BNB_CONFIG,
    dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    device_map={"": 0} if DEVICE != "cpu" else {"": "cpu"},  # Explicit device mapping for accelerate
)

print(f"Model loaded on device: {next(model.parameters()).device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Specific to OPT architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 7. Trainer Setup
training_args = SFTConfig(
    output_dir=MODEL_CONFIG.output_dir,
    per_device_train_batch_size=2 if DEVICE != "cpu" else 1,  # Smaller batch for stability
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,  # 1 epoch is enough to see the behavior change in 125M
    logging_steps=10,
    fp16=False,  # Disable fp16 to avoid device conflicts
    save_strategy="no",
    report_to="none",
    max_length=512,  # Changed from max_seq_length to max_length
    # Device-specific configurations
    dataloader_pin_memory=False,  # Avoid memory pinning issues
    dataloader_num_workers=0,  # Single threaded to avoid device conflicts
    remove_unused_columns=False,  # Keep all columns for stability
)

try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,  # Dataset is now preprocessed with formatting
        peft_config=peft_config,
        processing_class=tokenizer,  # Changed from tokenizer=tokenizer
        args=training_args,
    )
    print("✅ SFTTrainer created successfully")

    # Print device information for debugging
    print(f"Training on device: {DEVICE}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"FP16 enabled: {training_args.fp16}")
    print("Starting training...")

    # Train and Save
    trainer.train()
    trainer.model.save_pretrained("./opt-125m-adapter")
    print("Fine-tuning complete. Adapter saved.")

except Exception as e:
    print(f"❌ Error during training: {e}")
    print("This might be due to device compatibility issues.")
    print("Try uncommenting the 'DEVICE = \"cpu\"' line above to force CPU training.")
    raise e
