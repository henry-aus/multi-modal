from transformers import BitsAndBytesConfig
import torch

class ModelConfig:
    model_id = "facebook/opt-125m"
    dataset_id = "tatsu-lab/alpaca"
    output_dir = "./output"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_CONFIG = ModelConfig()