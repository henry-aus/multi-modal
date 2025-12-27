# Code Alpaca - Instruction Fine-tuning Project

A fine-tuning project for creating instruction-following language models using the Alpaca dataset with efficient training techniques.

## 🎯 Project Overview

This project implements supervised fine-tuning (SFT) of language models using:
- **Base Model**: Facebook OPT-125M
- **Dataset**: Stanford Alpaca instruction-following dataset
- **Training Method**: Parameter Efficient Fine-Tuning (PEFT) with LoRA
- **Quantization**: 4-bit quantization using BitsAndBytes
- **Device Support**: Cross-platform (CUDA, MPS, CPU) with automatic device detection

## 📁 Project Structure

```
code-alpaca/
├── train.py              # Main training script
├── compare.py             # Model comparison (before/after fine-tuning)
├── config.py              # Model and training configuration
├── device_utils.py        # Cross-platform device detection
├── requirements.txt       # Python dependencies
├── venv/                 # Python virtual environment
├── output/               # Training outputs and checkpoints
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create a Python 3.12 virtual environment with all dependencies:

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

Start the instruction fine-tuning process:

```bash
python train.py
```

The training will automatically:
- Detect the optimal device (CUDA/MPS/CPU)
- Load and quantize the OPT-125M model
- Apply LoRA adapters for efficient training
- Fine-tune on the Alpaca dataset

### Training Progress

![Training Progress](images/training-progress.png)
*Training successfully running on MPS device with optimized settings*

**Sample Training Output:**
```
Using device: mps
Model loaded on device: mps:0
✅ SFTTrainer created successfully
Training on device: mps
Model device: mps:0
FP16 enabled: False
Starting training...
{'loss': 2.9844, 'grad_norm': 2.1440625, 'learning_rate': 0.00019972311952007386,
 'epoch': 0.0, 'entropy': 3.4995821416373802, 'num_tokens': 6734.0}
  0%|          | 12/6501 [00:38<5:41:23,  3.16s/it]]
```

**Training Features:**
- ✅ **Device**: Automatic MPS/CUDA/CPU detection
- ✅ **Model**: OPT-125M with 4-bit quantization
- ✅ **Memory Efficient**: LoRA adapters (r=16, α=32)
- ✅ **Dataset**: Alpaca instruction-response pairs
- ✅ **Progress**: Real-time loss and metrics tracking

### 3. Model Comparison

Compare model performance before and after fine-tuning:

```bash
python compare.py
```

This will generate responses from both the base model and fine-tuned model for comparison.

## 🔧 Configuration

### Model Configuration (`config.py`)

```python
class ModelConfig:
    model_id = "facebook/opt-125m"      # Base model
    dataset_id = "tatsu-lab/alpaca"     # Training dataset
    output_dir = "./output"             # Output directory
```

### Training Parameters

- **Batch Size**: 2-4 (device-dependent)
- **Learning Rate**: 2e-4
- **Epochs**: 1 (sufficient for 125M model)
- **Max Length**: 512 tokens
- **LoRA Rank**: 16
- **Target Modules**: q_proj, v_proj (OPT-specific)

## 💻 Device Support

The project automatically detects and uses the best available device:

| Device | Status | Notes |
|--------|--------|-------|
| **CUDA** | ✅ Supported | NVIDIA GPUs with CUDA |
| **MPS** | ✅ Supported | Apple Silicon (M1/M2/M3) |
| **CPU** | ✅ Supported | Fallback option |

Device detection is handled by `device_utils.py` with automatic optimization.

## 📊 Training Details

### Memory Optimization
- **4-bit Quantization**: Reduces memory usage by ~75%
- **LoRA Adapters**: Train only 0.1% of model parameters
- **Gradient Accumulation**: Effective batch size scaling
- **Mixed Precision**: Disabled for device compatibility

### Dataset Processing
- **Format**: Alpaca instruction-response format
- **Template**: `### Instruction:\n{instruction}\n\n### Response:\n{response}`
- **Preprocessing**: Automatic tokenization and formatting

## 🛠️ Dependencies

Core dependencies include:
- `torch>=2.1.0` - PyTorch framework
- `transformers>=4.35.0` - Hugging Face transformers
- `trl>=0.7.0` - Transformer Reinforcement Learning
- `peft>=0.6.0` - Parameter Efficient Fine-Tuning
- `datasets>=2.14.0` - Hugging Face datasets
- `bitsandbytes>=0.41.0` - Quantization support

See `requirements.txt` for complete dependency list.

## 🔍 Troubleshooting

### Common Issues

**Device Errors**
```bash
# Force CPU training if device issues occur
# Uncomment this line in train.py:
# DEVICE = "cpu"
```

**Memory Issues**
- Reduce `per_device_train_batch_size` in training config
- Enable CPU offloading if needed
- Use smaller max_length (e.g., 256)

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📈 Results

After training, you'll have:
- **Fine-tuned Adapter**: `./opt-125m-adapter/` directory
- **Improved Responses**: Better instruction-following capability
- **Comparison Tool**: Side-by-side before/after evaluation

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Share your fine-tuning results

## 📄 License

This project is for educational and research purposes. Please respect the licenses of:
- Base model (Facebook OPT)
- Alpaca dataset (Stanford)
- Dependencies (see individual package licenses)

## 🔗 References

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Facebook OPT](https://huggingface.co/facebook/opt-125m)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [TRL Library](https://github.com/huggingface/trl)

## 📸 Adding Training Screenshots

To include your training progress screenshot:

1. Save your training output screenshot as `training-progress.png`
2. Place it in the `images/` directory
3. The README.md will automatically display it in the Training Progress section

The screenshot should capture the console output showing device detection, model loading, and training progress with metrics.

---

**Happy Fine-tuning!** 🎉

For questions or issues, please check the troubleshooting section or create an issue in the repository.