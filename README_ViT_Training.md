# Vision Transformer (ViT) Training and XAI Pipeline

A comprehensive pipeline for training Vision Transformer models on image classification tasks with Hugging Face-style model saving and post-hoc explainability analysis using Grad-CAM and Grad-CAM++.

## Features

- **Data Pipeline**: Automated image loading, preprocessing, and train/validation/test splitting
- **ViT Model**: Fine-tuning pretrained Vision Transformer models from Hugging Face
- **Training**: Complete training pipeline with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
- **HF-Style Saving**: Save models in Hugging Face format with config.json, weights, and label mappings
- **Comprehensive Evaluation**: Detailed metrics calculation and visualization
- **XAI Analysis**: Grad-CAM and Grad-CAM++ implementations for model interpretability

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (recommended), ensure you have CUDA installed and use:
```bash
pip install tensorflow-gpu>=2.12.0
```

## Dataset Structure

Your dataset should be organized in the following structure:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Usage

### 1. Training a ViT Model

#### Basic Usage
```bash
python vit_trainer.py
```

#### Customized Training
```python
from vit_trainer import ViTTrainer

# Initialize trainer with custom parameters
trainer = ViTTrainer(
    dataset_path="./your-dataset",
    model_name="google/vit-base-patch16-224",  # or "google/vit-large-patch16-224"
    image_size=224,
    batch_size=16,
    num_epochs=30,
    learning_rate=2e-5,
    output_dir="./model_output"
)

# Run the complete pipeline
trainer.run_full_pipeline()
```

#### Available Pretrained Models
- `google/vit-base-patch16-224` (Base model, recommended)
- `google/vit-large-patch16-224` (Large model, better performance but slower)
- `google/vit-base-patch32-224` (Faster training, slightly lower accuracy)

### 2. Model Outputs

After training, the following files will be created in your output directory:

```
vit_model_output/
├── model/
│   ├── config.json              # Model configuration
│   ├── label_map.json           # Class labels mapping
│   ├── tf_model.h5             # Model weights
│   ├── saved_model/            # SavedModel format
│   └── training_history.json   # Training metrics
├── evaluation_results.json     # Detailed evaluation metrics
├── evaluation_results.csv      # Evaluation metrics in CSV
├── confusion_matrix.png        # Confusion matrix visualization
├── training_history.png        # Training curves
└── best_model_weights.h5       # Best model weights during training
```

### 3. XAI Analysis with Grad-CAM

#### Command Line Usage

**Single Image Analysis:**
```bash
python gradcam_xai.py --model_dir ./vit_model_output/model --image path/to/image.jpg --output_dir ./gradcam_results
```

**Batch Processing:**
```bash
python gradcam_xai.py --model_dir ./vit_model_output/model --image_dir path/to/images/ --output_dir ./gradcam_results
```

**Using Grad-CAM++:**
```bash
python gradcam_xai.py --model_dir ./vit_model_output/model --image path/to/image.jpg --method gradcam++ --output_dir ./gradcam_results
```

**Target Specific Class:**
```bash
python gradcam_xai.py --model_dir ./vit_model_output/model --image path/to/image.jpg --target_class 2 --output_dir ./gradcam_results
```

#### Programmatic Usage

```python
from gradcam_xai import ViTGradCAM

# Initialize Grad-CAM analyzer
analyzer = ViTGradCAM("./vit_model_output/model")

# Single image analysis
analyzer.visualize_gradcam(
    image_path="path/to/image.jpg",
    save_path="./gradcam_output.png",
    use_gradcam_plus_plus=False,  # Set to True for Grad-CAM++
    target_class=None  # None for predicted class, or specify class index
)

# Batch processing
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
analyzer.batch_visualize(
    image_paths=image_paths,
    output_dir="./gradcam_batch_output",
    use_gradcam_plus_plus=False
)
```

## Configuration

### Training Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `dataset_path` | Path to dataset directory | `"./kvasir-dataset"` | - |
| `model_name` | Pretrained ViT model | `"google/vit-base-patch16-224"` | See available models above |
| `image_size` | Input image size | `224` | `224`, `384` |
| `batch_size` | Training batch size | `16` | `8-32` (depends on GPU memory) |
| `num_epochs` | Maximum training epochs | `30` | `20-100` |
| `learning_rate` | Learning rate | `2e-5` | `1e-5` to `5e-5` |
| `output_dir` | Output directory | `"./vit_model_output"` | - |

### Model Configuration

The saved `config.json` includes:
- Model architecture parameters
- Training hyperparameters
- Image preprocessing settings
- Layer names for XAI compatibility

## Model Performance

### Evaluation Metrics

The pipeline automatically calculates and saves:
- **Overall Metrics**: Accuracy, Precision, Recall, F1-Score
- **Per-Class Metrics**: Individual class performance
- **Confusion Matrix**: Visual representation of predictions
- **Training Curves**: Loss and accuracy progression

### Expected Performance

With the Kvasir dataset (8 classes, 4000 images):
- **Training Time**: ~30-60 minutes (with GPU)
- **Expected Accuracy**: 85-95% (depends on model size and training epochs)
- **Memory Usage**: 4-8GB GPU memory (batch_size=16)

## XAI Interpretability

### Grad-CAM vs Grad-CAM++

- **Grad-CAM**: Standard gradient-based visualization, faster computation
- **Grad-CAM++**: Improved version with better localization, slightly slower

### Visualization Outputs

Each XAI analysis produces:
1. **Original Image**: Input image
2. **Heatmap**: Attention visualization
3. **Superimposed**: Heatmap overlaid on original image

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `batch_size` (try 8 or 4)
   - Use smaller model variant
   - Reduce `image_size` to 224

2. **Slow Training**
   - Ensure GPU is available and being used
   - Check CUDA installation
   - Reduce dataset size for testing

3. **Model Loading Issues**
   - Ensure all required files are in model directory
   - Check file permissions
   - Verify TensorFlow and Transformers versions

4. **XAI Visualization Issues**
   - Ensure model was saved properly
   - Check image file formats (supported: jpg, png, bmp)
   - Verify model directory structure

### Performance Optimization

1. **For Better Accuracy**:
   - Use larger model (`vit-large-patch16-224`)
   - Increase `num_epochs`
   - Use data augmentation
   - Fine-tune learning rate

2. **For Faster Training**:
   - Use smaller model (`vit-base-patch32-224`)
   - Reduce `image_size` to 224
   - Increase `batch_size` if memory allows
   - Use mixed precision training

## Advanced Usage

### Custom Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add to ViTTrainer class
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

### Transfer Learning from Custom Models

```python
# Use your own pretrained model
trainer = ViTTrainer(
    dataset_path="./your-dataset",
    model_name="path/to/your/pretrained/model",
    # ... other parameters
)
```

### Custom XAI Layer Selection

```python
# Specify custom layer for Grad-CAM
analyzer = ViTGradCAM("./model_dir")
heatmap = analyzer.make_gradcam_heatmap(
    img_array,
    layer_name="vit.encoder.layer.11"  # Specific transformer layer
)
```

## File Structure

```
project/
├── vit_trainer.py              # Main training script
├── gradcam_xai.py             # XAI analysis script
├── requirements.txt           # Dependencies
├── README_ViT_Training.md     # This file
└── examples/
    ├── train_example.py       # Training example
    └── xai_example.py         # XAI analysis example
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vit_training_xai_pipeline,
  title={Vision Transformer Training and XAI Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/vit-training-xai}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section above
- Open an issue on GitHub
- Check the Hugging Face Transformers documentation
- Review TensorFlow documentation for GPU setup