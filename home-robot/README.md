# Home Robot Multi-Modal AI System

A comprehensive multi-modal AI system for home robotics applications, featuring object detection with GroundingDino and visual question answering capabilities using the NYU Depth V2 dataset.

## 🚀 Features

### Object Detection & Grounding
- **Text-Grounded Object Detection** using GroundingDino-Tiny
- **Natural Language Queries** for object detection ("find the chair", "locate kitchen items")
- **Bounding Box Prediction** with normalized coordinates
- **Multi-Object Detection** in indoor scenes

### Visual Question Answering (VQA)
- **Scene Understanding** with room classification
- **Interactive Q&A** about visual content
- **Context-Aware Responses** based on image analysis

### Dataset Processing
- **NYU Depth V2 Integration** with 1,449 indoor scenes
- **Automated Data Preprocessing** with robust error handling
- **Dynamic Text Prompts** optimized for model token limits
- **Binary Label Classification** for GroundingDino compatibility

## 📊 Training Demo

![Training Demo](training_demo.png)

*Successful GroundingDino training showing stable loss convergence with 172M parameters on CPU*

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- PyTorch 2.0+
- Transformers 4.30+

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd home-robot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
Download the NYU Depth V2 dataset and place `nyu_depth_v2_labeled.mat` in the `data/` directory.

## 🎯 Usage

### Quick Start - Dataset Demo
```bash
python data/nyu_depth_v2.py
```

### Training GroundingDino
```bash
# Basic training
python train/dino.py

# Custom parameters
python train/dino.py --max_epochs 20 --patience 10 --val_ratio 0.2
```

### Training Configuration
- **Learning Rate**: 1e-6 (conservative for stability)
- **Optimizer**: AdamW with weight decay 1e-4
- **Device**: CPU (MPS fallback for Apple Silicon compatibility)
- **Batch Processing**: Individual sample processing with robust error handling
- **Early Stopping**: Patience-based with best model restoration

## 📁 Project Structure

```
home-robot/
├── data/
│   ├── nyu_depth_v2.py          # Dataset processing and utilities
│   └── nyu_depth_v2_labeled.mat # NYU Depth V2 dataset (download required)
├── train/
│   ├── bclip.py                 # BLIP2 VQA training with LoRA fine-tuning
│   ├── dino.py                  # GroundingDino training script
│   └── training_utils.py        # Training utilities and metrics
├── device_utils.py              # Device detection and compatibility
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: IDEA-Research/grounding-dino-tiny
- **Parameters**: 172,249,090 (all trainable)
- **Input Resolution**: Dynamic (up to 800x1066)
- **Output**: Bounding boxes + confidence scores

### Data Processing Pipeline
1. **Instance Mask Processing**: Extract object instances from segmentation masks
2. **Bounding Box Generation**: Calculate normalized coordinates [0,1]
3. **Text Prompt Creation**: Generate class-specific prompts within token limits
4. **Binary Label Assignment**: Map to GroundingDino's binary classification (0=background, 1=object)
5. **Tensor Validation**: Comprehensive NaN/inf checking and data sanitization

### Performance Optimizations
- **Token Limit Management**: Dynamic text prompt generation (400 tokens max)
- **Memory Efficiency**: Individual sample processing to prevent OOM
- **Gradient Clipping**: Max norm 1.0 for training stability
- **Device Compatibility**: Automatic CPU fallback for unsupported operations

## 📈 Training Results

### Dataset Statistics
- **Total Samples**: 1,449 indoor scenes
- **Training Split**: 1,160 samples (80%)
- **Validation Split**: 289 samples (20%)
- **Object Classes**: 894 unique categories
- **Average Objects per Scene**: 5-10 instances

### Training Metrics
- **Loss Components**:
  - Classification Loss (loss_ce): ~1-3
  - Bounding Box Loss (loss_bbox): ~0.5-0.9
  - GIoU Loss (loss_giou): ~0.7-1.0
- **Training Speed**: ~6.8s/iteration on CPU
- **Convergence**: Stable loss progression with early stopping

## 🏠 Applications

### Home Robotics Use Cases
- **Object Search**: "Find my keys", "Locate the remote control"
- **Room Navigation**: Visual scene understanding for autonomous movement
- **Inventory Management**: Track household items and their locations
- **Safety Monitoring**: Detect hazards or unusual objects
- **Interactive Assistance**: Answer questions about visual environment

### Integration Examples
```python
# Object Detection Example
from data.nyu_depth_v2 import NYURobotDataset
from transformers import GroundingDinoProcessor

processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dataset = NYURobotDataset(processor, task="grounding")

# Get detection results
inputs, target = dataset[0]
# Process with trained model...

# VQA Example
vqa_dataset = NYURobotDataset(processor, task="vqa")
image, question, answer = vqa_dataset[0]
print(f"Q: {question}")
print(f"A: {answer}")
```

## 🔍 Troubleshooting

### Common Issues

**Device Compatibility**
- The system automatically falls back to CPU on Apple Silicon for GroundingDino compatibility
- CUDA GPUs are supported if available

**Memory Issues**
- Individual sample processing prevents batch-related OOM errors
- Reduce dataset size or increase system RAM if needed

**Training Instability**
- Conservative learning rate (1e-6) ensures stable training
- Gradient clipping prevents exploding gradients
- Early stopping prevents overfitting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NYU Depth V2 Dataset**: Silberman et al. for the comprehensive indoor scene dataset
- **GroundingDino**: IDEA Research for the text-grounded object detection model
- **Hugging Face Transformers**: For the model implementation and utilities
- **PyTorch**: For the deep learning framework

## 📚 References

- [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [GroundingDino Paper](https://arxiv.org/abs/2303.05499)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Built with ❤️ for home robotics and computer vision applications**