#!/usr/bin/env python3
"""
Grad-CAM and Grad-CAM++ Implementation for Vision Transformer (ViT) Models
Post-hoc XAI analysis for saved Hugging Face-style models.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
import argparse
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ViTGradCAM:
    """Grad-CAM and Grad-CAM++ implementation for Vision Transformer models."""
    
    def __init__(self, model_dir: str):
        """
        Initialize ViTGradCAM with a saved model.
        
        Args:
            model_dir: Path to the saved model directory containing config.json and model files
        """
        self.model_dir = model_dir
        self.model = None
        self.config = None
        self.label_map = None
        
        # Load model and configuration
        self.load_model_and_config()
        
    def load_model_and_config(self):
        """Load the saved model and its configuration."""
        print(f"Loading model from: {self.model_dir}")
        
        # Load configuration
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("Configuration loaded successfully")
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        # Load label map
        label_map_path = os.path.join(self.model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print("Label map loaded successfully")
        else:
            raise FileNotFoundError(f"Label map not found at {label_map_path}")
        
        # Try to load SavedModel format first, then fall back to weights
        saved_model_path = os.path.join(self.model_dir, "saved_model")
        if os.path.exists(saved_model_path):
            try:
                self.model = tf.keras.models.load_model(saved_model_path)
                print("Model loaded from SavedModel format")
            except Exception as e:
                print(f"Error loading SavedModel: {e}")
                print("Attempting to load from weights...")
                self._load_from_weights()
        else:
            self._load_from_weights()
    
    def _load_from_weights(self):
        """Load model from weights file."""
        try:
            from transformers import TFViTForImageClassification, ViTConfig
            
            # Recreate model architecture
            config = ViTConfig.from_dict(self.config)
            self.model = TFViTForImageClassification(config)
            
            # Build model by calling it once
            dummy_input = tf.random.normal((1, self.config["image_size"][0], 
                                          self.config["image_size"][1], 
                                          self.config["image_size"][2]))
            _ = self.model(dummy_input)
            
            # Load weights
            weights_path = os.path.join(self.model_dir, "tf_model.h5")
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                print("Model weights loaded successfully")
            else:
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from weights: {e}")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for ViT input.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image, preprocessed_image)
        """
        # Load and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()
        
        # Resize to model input size
        img_size = self.config["image_size"][:2]  # [height, width]
        img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return original_img, img_batch
    
    def get_last_conv_layer_name(self) -> str:
        """
        Find the last convolutional/transformer layer for Grad-CAM.
        For ViT, we typically use the last encoder layer.
        """
        # For ViT models, we want the last transformer layer
        # This is typically in the encoder
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):  # Check if it's a container layer
                for sublayer in reversed(layer.layers):
                    if 'encoder' in sublayer.name.lower() or 'transformer' in sublayer.name.lower():
                        return sublayer.name
            elif 'encoder' in layer.name.lower() or 'transformer' in layer.name.lower():
                return layer.name
        
        # Fallback: use the layer specified in config
        if "last_layer_name" in self.config:
            return self.config["last_layer_name"]
        
        # Final fallback: use the second-to-last layer
        return self.model.layers[-2].name
    
    def make_gradcam_heatmap(self, 
                           img_array: np.ndarray, 
                           pred_index: Optional[int] = None,
                           layer_name: Optional[str] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_array: Preprocessed image array
            pred_index: Index of the class to generate heatmap for (None for predicted class)
            layer_name: Name of the layer to use for Grad-CAM (None for auto-detection)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        if layer_name is None:
            layer_name = self.get_last_conv_layer_name()
        
        print(f"Using layer: {layer_name}")
        
        # Create a model that maps the input image to the activations of the target layer
        # as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the target layer
        with tf.GradientTape() as tape:
            layer_output, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, layer_output)
        
        # For ViT, we need to handle the transformer output differently
        # The output is typically (batch_size, num_patches + 1, hidden_size)
        # We need to exclude the CLS token and reshape to spatial dimensions
        
        if len(layer_output.shape) == 3:  # Transformer output
            # Remove batch dimension and CLS token (first token)
            layer_output = layer_output[0, 1:]  # (num_patches, hidden_size)
            grads = grads[0, 1:]  # (num_patches, hidden_size)
            
            # Calculate the patch size and spatial dimensions
            patch_size = self.config.get("patch_size", 16)
            img_size = self.config["image_size"][0]  # Assuming square images
            num_patches_per_side = img_size // patch_size
            
            # Compute the mean of gradients across the feature dimension
            pooled_grads = tf.reduce_mean(grads, axis=-1)  # (num_patches,)
            
            # Weight the layer output by the gradients
            layer_output = tf.multiply(pooled_grads[:, tf.newaxis], layer_output)
            
            # Sum across the feature dimension to get the heatmap
            heatmap = tf.reduce_sum(layer_output, axis=-1)  # (num_patches,)
            
            # Reshape to spatial dimensions
            heatmap = tf.reshape(heatmap, (num_patches_per_side, num_patches_per_side))
            
        else:  # Regular convolutional output
            # This is the standard Grad-CAM computation
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            layer_output = layer_output[0]
            heatmap = layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def make_gradcam_plus_plus_heatmap(self, 
                                     img_array: np.ndarray, 
                                     pred_index: Optional[int] = None,
                                     layer_name: Optional[str] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap (improved version of Grad-CAM).
        
        Args:
            img_array: Preprocessed image array
            pred_index: Index of the class to generate heatmap for (None for predicted class)
            layer_name: Name of the layer to use for Grad-CAM++ (None for auto-detection)
            
        Returns:
            Grad-CAM++ heatmap as numpy array
        """
        if layer_name is None:
            layer_name = self.get_last_conv_layer_name()
        
        print(f"Using layer for Grad-CAM++: {layer_name}")
        
        # Create a model that maps the input image to the activations of the target layer
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    layer_output, predictions = grad_model(img_array)
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]
                
                # First-order gradients
                grads = tape3.gradient(class_channel, layer_output)
            
            # Second-order gradients
            grads2 = tape2.gradient(grads, layer_output)
        
        # Third-order gradients
        grads3 = tape1.gradient(grads2, layer_output)
        
        # Handle transformer output similar to regular Grad-CAM
        if len(layer_output.shape) == 3:  # Transformer output
            layer_output = layer_output[0, 1:]  # Remove batch dim and CLS token
            grads = grads[0, 1:]
            grads2 = grads2[0, 1:]
            grads3 = grads3[0, 1:]
            
            # Calculate alpha weights for Grad-CAM++
            alpha = grads2 / (2.0 * grads2 + tf.reduce_sum(layer_output * grads3, axis=-1, keepdims=True) + 1e-7)
            
            # Compute weighted combination
            weights = tf.maximum(grads, 0) * alpha
            weights = tf.reduce_sum(weights, axis=-1)
            
            # Weight the layer output
            weighted_output = weights[:, tf.newaxis] * layer_output
            heatmap = tf.reduce_sum(weighted_output, axis=-1)
            
            # Reshape to spatial dimensions
            patch_size = self.config.get("patch_size", 16)
            img_size = self.config["image_size"][0]
            num_patches_per_side = img_size // patch_size
            heatmap = tf.reshape(heatmap, (num_patches_per_side, num_patches_per_side))
            
        else:  # Regular convolutional output
            layer_output = layer_output[0]
            
            # Calculate alpha weights
            alpha = grads2[0] / (2.0 * grads2[0] + tf.reduce_sum(layer_output * grads3[0], axis=(0, 1), keepdims=True) + 1e-7)
            
            # Compute weighted combination
            weights = tf.reduce_sum(tf.maximum(grads[0], 0) * alpha, axis=(0, 1))
            heatmap = tf.reduce_sum(weights * layer_output, axis=-1)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-7)
        
        return heatmap.numpy()
    
    def visualize_gradcam(self, 
                         image_path: str, 
                         save_path: Optional[str] = None,
                         use_gradcam_plus_plus: bool = False,
                         target_class: Optional[int] = None) -> None:
        """
        Generate and visualize Grad-CAM heatmap for an image.
        
        Args:
            image_path: Path to the input image
            save_path: Path to save the visualization (None to display only)
            use_gradcam_plus_plus: Whether to use Grad-CAM++ instead of Grad-CAM
            target_class: Target class index (None for predicted class)
        """
        # Preprocess image
        original_img, img_array = self.preprocess_image(image_path)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(tf.nn.softmax(predictions[0]))
        
        # Use target class or predicted class
        class_idx = target_class if target_class is not None else predicted_class
        
        # Generate heatmap
        method_name = "Grad-CAM++" if use_gradcam_plus_plus else "Grad-CAM"
        print(f"Generating {method_name} for class: {self.label_map['id2label'][str(class_idx)]}")
        
        if use_gradcam_plus_plus:
            heatmap = self.make_gradcam_plus_plus_heatmap(img_array, class_idx)
        else:
            heatmap = self.make_gradcam_heatmap(img_array, class_idx)
        
        # Resize heatmap to original image size
        img_size = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (img_size[1], img_size[0]))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title(f'{method_name} Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Superimposed
        superimposed = original_img * 0.6 + cm.jet(heatmap_resized)[:, :, :3] * 255 * 0.4
        superimposed = superimposed.astype(np.uint8)
        axes[2].imshow(superimposed)
        axes[2].set_title('Superimposed')
        axes[2].axis('off')
        
        # Add prediction info
        pred_label = self.label_map['id2label'][str(predicted_class)]
        target_label = self.label_map['id2label'][str(class_idx)]
        
        fig.suptitle(f'Predicted: {pred_label} (confidence: {confidence:.3f})\n'
                    f'Target Class: {target_label}', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def batch_visualize(self, 
                       image_paths: List[str], 
                       output_dir: str,
                       use_gradcam_plus_plus: bool = False) -> None:
        """
        Generate Grad-CAM visualizations for multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save visualizations
            use_gradcam_plus_plus: Whether to use Grad-CAM++ instead of Grad-CAM
        """
        os.makedirs(output_dir, exist_ok=True)
        method_name = "gradcam_plus_plus" if use_gradcam_plus_plus else "gradcam"
        
        for i, image_path in enumerate(image_paths):
            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(output_dir, f"{base_name}_{method_name}.png")
                
                print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
                self.visualize_gradcam(image_path, save_path, use_gradcam_plus_plus)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Batch processing completed. Results saved to: {output_dir}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Grad-CAM/Grad-CAM++ Analysis for ViT Models")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to the saved model directory")
    parser.add_argument("--image", type=str,
                       help="Path to a single image for analysis")
    parser.add_argument("--image_dir", type=str,
                       help="Directory containing images for batch processing")
    parser.add_argument("--output_dir", type=str, default="./gradcam_output",
                       help="Output directory for visualizations")
    parser.add_argument("--method", type=str, choices=["gradcam", "gradcam++"], 
                       default="gradcam", help="XAI method to use")
    parser.add_argument("--target_class", type=int, default=None,
                       help="Target class index (default: use predicted class)")
    
    args = parser.parse_args()
    
    # Initialize Grad-CAM analyzer
    analyzer = ViTGradCAM(args.model_dir)
    
    use_gradcam_plus_plus = (args.method == "gradcam++")
    
    if args.image:
        # Single image analysis
        save_path = os.path.join(args.output_dir, 
                                f"{os.path.splitext(os.path.basename(args.image))[0]}_{args.method}.png")
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.visualize_gradcam(args.image, save_path, use_gradcam_plus_plus, args.target_class)
        
    elif args.image_dir:
        # Batch processing
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if image_paths:
            analyzer.batch_visualize(image_paths, args.output_dir, use_gradcam_plus_plus)
        else:
            print(f"No images found in {args.image_dir}")
    
    else:
        print("Please provide either --image or --image_dir")
        return


if __name__ == "__main__":
    main()