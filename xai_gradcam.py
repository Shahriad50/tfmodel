import os
import json
import argparse
from typing import Tuple

import numpy as np
import cv2
import tensorflow as tf
from transformers import TFAutoModelForImageClassification, AutoImageProcessor


def load_model(bundle_dir: str):
	with open(os.path.join(bundle_dir, "config.json"), "r", encoding="utf-8") as f:
		config = json.load(f)
	processor_name = config.get("processor_name", "google/vit-base-patch16-224")
	id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
	label2id = {k: int(v) for k, v in config.get("label2id", {}).items()}
	model = TFAutoModelForImageClassification.from_pretrained(processor_name, num_labels=len(id2label) or None, id2label=id2label or None, label2id=label2id or None)
	# Load fine-tuned weights
	weights_path = os.path.join(bundle_dir, "tf_model.h5")
	if os.path.exists(weights_path):
		model.load_weights(weights_path)
	processor = AutoImageProcessor.from_pretrained(processor_name)
	image_size = tuple(config.get("image_size", [224, 224, 3]))
	last_layer_name = config.get("last_layer_name", None)
	return model, processor, image_size, last_layer_name, id2label


def preprocess_image(img_path: str, image_size: Tuple[int, int]) -> tf.Tensor:
	img = tf.io.read_file(img_path)
	img = tf.image.decode_image(img, channels=3, expand_animations=False)
	img = tf.image.resize(img, image_size[:2])
	img = tf.cast(img, tf.float32) / 255.0
	return img


def compute_gradcam(model: tf.keras.Model, image_tensor: tf.Tensor, last_layer_name: str, use_plus: bool = False) -> np.ndarray:
	# Expand dims to batch size 1
	input_tensor = tf.expand_dims(image_tensor, axis=0)
	# Try to get the specified layer; if not present, fall back to penultimate layer
	try:
		last_layer = model.get_layer(last_layer_name)
	except Exception:
		last_layer = model.layers[-2]

	# If the chosen layer isn't conv-like (ViT), we still compute saliency over the last hidden feature map if available
	try:
		grad_model = tf.keras.Model([model.inputs], [last_layer.output, model.output])
	except Exception:
		grad_model = tf.keras.Model([model.inputs], [model.output])

	with tf.GradientTape() as tape:
		outputs = grad_model(input_tensor)
		if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
			feature_map, predictions = outputs
		else:
			feature_map, predictions = None, outputs
		class_idx = tf.argmax(predictions[0])
		loss = predictions[:, class_idx]
	if feature_map is None:
		# Saliency map fallback: gradient of score w.r.t input
		with tf.GradientTape() as tape2:
			tape2.watch(input_tensor)
			predictions2 = model(input_tensor, training=False)
			if hasattr(predictions2, 'logits'):
				predictions2 = predictions2.logits
			class_idx2 = tf.argmax(predictions2[0])
			loss2 = predictions2[:, class_idx2]
		grads2 = tape2.gradient(loss2, input_tensor)[0]
		gradcam = tf.reduce_max(tf.abs(grads2), axis=-1)
		gradcam -= tf.reduce_min(gradcam)
		gradcam /= (tf.reduce_max(gradcam) + 1e-8)
		return gradcam.numpy()

	# Standard Grad-CAM path
	grads = tape.gradient(loss, feature_map)
	if use_plus:
		squared_grads = tf.square(grads)
		relu_grads = tf.nn.relu(grads)
		alpha_num = squared_grads
		alpha_denom = 2.0 * squared_grads + tf.reduce_sum(feature_map * tf.pow(grads, 3), axis=(1, 2), keepdims=True)
		alpha = alpha_num / (alpha_denom + 1e-7)
		weights = tf.reduce_sum(alpha * relu_grads, axis=(1, 2))
	else:
		weights = tf.reduce_mean(grads, axis=(1, 2))

	cam = tf.reduce_sum(weights[:, None, None, :] * feature_map, axis=-1)
	cam = tf.nn.relu(cam)
	cam = cam[0]
	cam -= tf.reduce_min(cam)
	cam /= (tf.reduce_max(cam) + 1e-8)
	cam = cam.numpy()
	cam = cv2.resize(cam, (image_tensor.shape[1], image_tensor.shape[0]))
	return cam


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
	heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
	heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
	overlay = (alpha * heatmap + (1 - alpha) * image_rgb).astype(np.uint8)
	return overlay


def run_xai(bundle_dir: str, image_path: str, out_path: str, use_plus: bool):
	model, processor, image_size, last_layer_name, id2label = load_model(bundle_dir)
	if last_layer_name is None:
		last_layer_name = model.layers[-1].name
	img = preprocess_image(image_path, image_size)
	cam = compute_gradcam(model, img, last_layer_name, use_plus=use_plus)
	img_np = (img.numpy() * 255).astype(np.uint8)
	overlay = overlay_heatmap(img_np, cam)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
	print(f"Saved heatmap to {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Grad-CAM/Grad-CAM++ for HF-style TF ViT model")
	parser.add_argument("--model_dir", type=str, required=True)
	parser.add_argument("--image", type=str, required=True)
	parser.add_argument("--out", type=str, required=True)
	parser.add_argument("--method", type=str, choices=["gradcam", "gradcam++"], default="gradcam")
	args = parser.parse_args()
	use_plus = args.method == "gradcam++"
	run_xai(args.model_dir, args.image, args.out, use_plus)


if __name__ == "__main__":
	main()