#!/usr/bin/env python3
"""
Vision Transformer (ViT) Training Pipeline for Image Classification
with Hugging Face-style model saving and XAI compatibility.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from transformers import TFViTForImageClassification, ViTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ViTTrainer:
    """Vision Transformer trainer with Hugging Face-style model saving."""
    
    def __init__(self, 
                 dataset_path: str,
                 model_name: str = "google/vit-base-patch16-224",
                 image_size: int = 224,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 learning_rate: float = 2e-5,
                 output_dir: str = "./vit_model_output"):
        
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        # Will be populated during data loading
        self.categories = []
        self.num_classes = 0
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_data_categories(self):
        """Extract categories from dataset directory structure."""
        categories = []
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if os.path.isdir(folder_path):
                # Count image files
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if image_files:  # Only add if folder contains images
                    categories.append((folder_name, len(image_files)))
        
        # Sort categories by name for consistency
        categories.sort(key=lambda x: x[0])
        self.categories = [cat[0] for cat in categories]
        self.num_classes = len(self.categories)
        
        print(f"Found {self.num_classes} categories: {self.categories}")
        return categories
    
    def load_and_preprocess_data(self):
        """Load images and create dataset with proper preprocessing for ViT."""
        print("Loading and preprocessing data...")
        
        categories_info = self.get_data_categories()
        
        X, y = [], []
        
        for category_name, _ in categories_info:
            category_path = os.path.join(self.dataset_path, category_name)
            class_idx = self.categories.index(category_name)
            
            print(f"Processing category: {category_name} (class {class_idx})")
            
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(category_path, img_file)
                    try:
                        # Load image
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Resize to target size
                            img = cv2.resize(img, (self.image_size, self.image_size))
                            
                            X.append(img)
                            y.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Normalize pixel values to [0, 1] range (ViT expects this)
        X = X / 255.0
        
        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        
        # Split into train/val/test (60/20/20)
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        print(f"Train set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def create_model(self):
        """Create and configure ViT model for fine-tuning."""
        print(f"Creating ViT model: {self.model_name}")
        
        # Create ViT configuration
        config = ViTConfig.from_pretrained(self.model_name)
        config.num_labels = self.num_classes
        config.id2label = {i: label for i, label in enumerate(self.categories)}
        config.label2id = {label: i for i, label in enumerate(self.categories)}
        
        # Create model
        self.model = TFViTForImageClassification.from_pretrained(
            self.model_name,
            config=config,
            from_tf=True
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        print("Model created and compiled successfully!")
        return self.model
    
    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model_weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self):
        """Train the ViT model."""
        print("Starting training...")
        
        callbacks = self.create_callbacks()
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def save_model_hf_style(self):
        """Save model in Hugging Face style with config, weights, and label map."""
        print("Saving model in Hugging Face style...")
        
        # Create model directory
        model_dir = os.path.join(self.output_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights in TensorFlow format
        tf_model_path = os.path.join(model_dir, "tf_model.h5")
        self.model.save_weights(tf_model_path)
        print(f"Model weights saved to: {tf_model_path}")
        
        # Also save as SavedModel format for better compatibility
        saved_model_path = os.path.join(model_dir, "saved_model")
        self.model.save(saved_model_path, save_format='tf')
        print(f"SavedModel saved to: {saved_model_path}")
        
        # Create and save config.json
        config_data = {
            "model_type": "vit",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "image_size": [self.image_size, self.image_size, 3],
            "patch_size": 16,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "intermediate_size": 3072,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "gradient_checkpointing": False,
            "last_layer_name": "classifier",  # For Grad-CAM
            "training_info": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs_trained": len(self.history.history['loss']) if self.history else 0,
                "training_date": datetime.now().isoformat()
            }
        }
        
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Config saved to: {config_path}")
        
        # Create and save label_map.json
        label_map = {
            "id2label": {i: label for i, label in enumerate(self.categories)},
            "label2id": {label: i for i, label in enumerate(self.categories)},
            "categories": self.categories,
            "num_classes": self.num_classes
        }
        
        label_map_path = os.path.join(model_dir, "label_map.json")
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        print(f"Label map saved to: {label_map_path}")
        
        # Save training history
        if self.history:
            history_path = os.path.join(model_dir, "training_history.json")
            history_data = {key: [float(val) for val in values] 
                          for key, values in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"Training history saved to: {history_path}")
        
        print("Model saved in Hugging Face style successfully!")
    
    def evaluate_model(self):
        """Evaluate model on test set and save comprehensive metrics."""
        print("Evaluating model on test set...")
        
        # Make predictions
        test_predictions = self.model.predict(self.X_test, batch_size=self.batch_size)
        test_pred_classes = np.argmax(test_predictions.logits, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, test_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, test_pred_classes, average='weighted'
        )
        
        # Classification report
        class_report = classification_report(
            self.y_test, test_pred_classes, 
            target_names=self.categories, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred_classes)
        
        # Prepare evaluation results
        evaluation_results = {
            "overall_metrics": {
                "accuracy": float(accuracy),
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "num_test_samples": len(self.y_test)
            },
            "per_class_metrics": {
                category: {
                    "precision": float(class_report[category]["precision"]),
                    "recall": float(class_report[category]["recall"]),
                    "f1-score": float(class_report[category]["f1-score"]),
                    "support": int(class_report[category]["support"])
                }
                for category in self.categories
            },
            "confusion_matrix": cm.tolist(),
            "categories": self.categories,
            "evaluation_date": datetime.now().isoformat()
        }
        
        # Save evaluation results as JSON
        eval_json_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(eval_json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Evaluation results saved to: {eval_json_path}")
        
        # Save evaluation results as CSV
        eval_csv_data = []
        for category in self.categories:
            eval_csv_data.append({
                "category": category,
                "precision": class_report[category]["precision"],
                "recall": class_report[category]["recall"],
                "f1_score": class_report[category]["f1-score"],
                "support": class_report[category]["support"]
            })
        
        eval_df = pd.DataFrame(eval_csv_data)
        eval_csv_path = os.path.join(self.output_dir, "evaluation_results.csv")
        eval_df.to_csv(eval_csv_path, index=False)
        print(f"Evaluation CSV saved to: {eval_csv_path}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.categories, yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to: {cm_path}")
        
        # Print summary
        print(f"\n=== Evaluation Summary ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return evaluation_results
    
    def plot_training_history(self):
        """Plot and save training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training & validation loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate if available
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Summary statistics
        max_val_acc = max(self.history.history['val_accuracy'])
        min_val_loss = min(self.history.history['val_loss'])
        axes[1, 1].text(0.1, 0.8, f'Best Validation Accuracy: {max_val_acc:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Best Validation Loss: {min_val_loss:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Total Epochs: {len(self.history.history["loss"])}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        history_plot_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {history_plot_path}")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print("=== Starting ViT Training Pipeline ===")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create model
        self.create_model()
        
        # Train model
        self.train_model()
        
        # Plot training history
        self.plot_training_history()
        
        # Save model in HF style
        self.save_model_hf_style()
        
        # Evaluate model
        self.evaluate_model()
        
        print("=== ViT Training Pipeline Completed Successfully! ===")
        print(f"All outputs saved to: {self.output_dir}")


def main():
    """Main function to run the ViT training pipeline."""
    
    # Configuration
    dataset_path = "./kvasir-dataset"  # Update this path as needed
    output_dir = "./vit_model_output"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' not found!")
        print("Please update the dataset_path variable or ensure the dataset is available.")
        return
    
    # Initialize trainer
    trainer = ViTTrainer(
        dataset_path=dataset_path,
        model_name="google/vit-base-patch16-224",
        image_size=224,
        batch_size=16,  # Reduced for better memory usage
        num_epochs=30,
        learning_rate=2e-5,
        output_dir=output_dir
    )
    
    # Run full pipeline
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()