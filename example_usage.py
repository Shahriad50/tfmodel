#!/usr/bin/env python3
"""
Example usage of the ViT Training and XAI Pipeline
This script demonstrates how to train a ViT model and perform XAI analysis.
"""

import os
from vit_trainer import ViTTrainer
from gradcam_xai import ViTGradCAM

def train_vit_model():
    """Example of training a ViT model."""
    print("=== ViT Model Training Example ===")
    
    # Configuration
    dataset_path = "./kvasir-dataset"  # Update this to your dataset path
    output_dir = "./vit_model_output"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please update the dataset_path or ensure your dataset is available.")
        print("Expected structure:")
        print("dataset/")
        print("├── class1/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("├── class2/")
        print("│   ├── image1.jpg")
        print("│   └── ...")
        print("└── ...")
        return None
    
    # Initialize trainer with custom parameters
    trainer = ViTTrainer(
        dataset_path=dataset_path,
        model_name="google/vit-base-patch16-224",
        image_size=224,
        batch_size=16,  # Adjust based on your GPU memory
        num_epochs=10,  # Reduced for example - increase for better performance
        learning_rate=2e-5,
        output_dir=output_dir
    )
    
    print(f"Training configuration:")
    print(f"- Dataset: {dataset_path}")
    print(f"- Model: {trainer.model_name}")
    print(f"- Image size: {trainer.image_size}")
    print(f"- Batch size: {trainer.batch_size}")
    print(f"- Epochs: {trainer.num_epochs}")
    print(f"- Learning rate: {trainer.learning_rate}")
    print(f"- Output directory: {output_dir}")
    
    # Run the complete training pipeline
    try:
        trainer.run_full_pipeline()
        print(f"Training completed successfully! Model saved to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None

def perform_xai_analysis(model_dir, sample_image_path=None):
    """Example of XAI analysis using Grad-CAM."""
    print("\n=== XAI Analysis Example ===")
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please train a model first or provide a valid model directory.")
        return
    
    # Initialize Grad-CAM analyzer
    try:
        analyzer = ViTGradCAM(model_dir)
        print("Grad-CAM analyzer initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize Grad-CAM analyzer: {e}")
        return
    
    # If no sample image provided, try to find one from the dataset
    if sample_image_path is None:
        # Try to find a sample image from the dataset
        dataset_path = "./kvasir-dataset"
        if os.path.exists(dataset_path):
            for class_dir in os.listdir(dataset_path):
                class_path = os.path.join(dataset_path, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            sample_image_path = os.path.join(class_path, img_file)
                            break
                    if sample_image_path:
                        break
    
    if sample_image_path is None or not os.path.exists(sample_image_path):
        print("No sample image found for XAI analysis.")
        print("Please provide a valid image path or ensure your dataset contains images.")
        return
    
    print(f"Using sample image: {sample_image_path}")
    
    # Create output directory for XAI results
    xai_output_dir = "./gradcam_results"
    os.makedirs(xai_output_dir, exist_ok=True)
    
    # Perform Grad-CAM analysis
    print("Performing Grad-CAM analysis...")
    try:
        # Standard Grad-CAM
        gradcam_save_path = os.path.join(xai_output_dir, "gradcam_example.png")
        analyzer.visualize_gradcam(
            image_path=sample_image_path,
            save_path=gradcam_save_path,
            use_gradcam_plus_plus=False
        )
        print(f"Grad-CAM visualization saved to: {gradcam_save_path}")
        
        # Grad-CAM++
        gradcam_pp_save_path = os.path.join(xai_output_dir, "gradcam_plus_plus_example.png")
        analyzer.visualize_gradcam(
            image_path=sample_image_path,
            save_path=gradcam_pp_save_path,
            use_gradcam_plus_plus=True
        )
        print(f"Grad-CAM++ visualization saved to: {gradcam_pp_save_path}")
        
    except Exception as e:
        print(f"XAI analysis failed with error: {e}")

def main():
    """Main function that demonstrates the complete pipeline."""
    print("ViT Training and XAI Pipeline Example")
    print("=" * 50)
    
    # Option 1: Train a new model
    print("\nOption 1: Train a new ViT model")
    train_new = input("Do you want to train a new model? (y/n): ").lower().strip()
    
    if train_new == 'y':
        model_dir = train_vit_model()
        if model_dir:
            model_dir = os.path.join(model_dir, "model")
        else:
            print("Training failed. Exiting.")
            return
    else:
        # Option 2: Use existing model
        print("\nOption 2: Use existing model")
        model_dir = input("Enter path to existing model directory (e.g., ./vit_model_output/model): ").strip()
        if not model_dir:
            model_dir = "./vit_model_output/model"
    
    # Perform XAI analysis
    print("\nPerforming XAI analysis...")
    sample_image = input("Enter path to sample image (press Enter to auto-find): ").strip()
    if not sample_image:
        sample_image = None
    
    perform_xai_analysis(model_dir, sample_image)
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Check the training results in the output directory")
    print("2. Review the XAI visualizations in ./gradcam_results/")
    print("3. Experiment with different images and XAI methods")
    print("4. Try batch processing for multiple images")

def quick_demo():
    """Quick demonstration without user input."""
    print("Quick Demo: ViT Training and XAI Pipeline")
    print("=" * 50)
    
    # Use default paths
    dataset_path = "./kvasir-dataset"
    model_output = "./vit_model_output"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure your dataset is available for the quick demo.")
        return
    
    # Quick training with minimal epochs
    print("Training ViT model (quick demo with 5 epochs)...")
    trainer = ViTTrainer(
        dataset_path=dataset_path,
        model_name="google/vit-base-patch16-224",
        image_size=224,
        batch_size=8,  # Small batch for demo
        num_epochs=5,   # Very few epochs for quick demo
        learning_rate=2e-5,
        output_dir=model_output
    )
    
    try:
        trainer.run_full_pipeline()
        print("Training completed!")
        
        # Quick XAI demo
        model_dir = os.path.join(model_output, "model")
        perform_xai_analysis(model_dir)
        
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ViT Training and XAI Pipeline Example")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo without user interaction")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    else:
        main()