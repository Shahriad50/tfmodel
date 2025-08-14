import os
import json
import math
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoImageProcessor, TFAutoModelForImageClassification


AUTOTUNE = tf.data.AUTOTUNE


def create_label_map_from_folders(dataset_dir: str) -> Dict[str, int]:
	class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
	label_map = {name: idx for idx, name in enumerate(class_names)}
	return label_map


def list_image_paths_and_labels(dataset_dir: str, label_map: Dict[str, int]) -> Tuple[List[str], List[int]]:
	image_paths: List[str] = []
	labels: List[int] = []
	for class_name, idx in label_map.items():
		class_dir = os.path.join(dataset_dir, class_name)
		for root, _, files in os.walk(class_dir):
			for f in files:
				if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
					image_paths.append(os.path.join(root, f))
					labels.append(idx)
	return image_paths, labels


def train_val_test_split(paths: List[str], labels: List[int], val_ratio: float, test_ratio: float, seed: int) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
	rng = np.random.default_rng(seed)
	indices = np.arange(len(paths))
	rng.shuffle(indices)
	paths = [paths[i] for i in indices]
	labels = [labels[i] for i in indices]
	
	n_total = len(paths)
	n_test = int(math.floor(n_total * test_ratio))
	n_val = int(math.floor(n_total * val_ratio))
	
	test_paths, test_labels = paths[:n_test], labels[:n_test]
	val_paths, val_labels = paths[n_test:n_test + n_val], labels[n_test:n_test + n_val]
	train_paths, train_labels = paths[n_test + n_val:], labels[n_test + n_val:]
	return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def build_tf_dataset(paths: List[str], labels: List[int], image_size: Tuple[int, int], batch_size: int, shuffle: bool, processor: AutoImageProcessor) -> tf.data.Dataset:
	img_height, img_width = image_size
	path_ds = tf.data.Dataset.from_tensor_slices(paths)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int32))
	ds = tf.data.Dataset.zip((path_ds, label_ds))

	image_mean = tf.constant(processor.image_mean if hasattr(processor, "image_mean") else [0.5, 0.5, 0.5], dtype=tf.float32)
	image_std = tf.constant(processor.image_std if hasattr(processor, "image_std") else [0.5, 0.5, 0.5], dtype=tf.float32)

	def _load_image(path: tf.Tensor) -> tf.Tensor:
		image = tf.io.read_file(path)
		image = tf.image.decode_image(image, channels=3, expand_animations=False)
		image.set_shape([None, None, 3])
		image = tf.image.resize(image, [img_height, img_width], method=tf.image.ResizeMethod.BILINEAR)
		image = tf.cast(image, tf.float32) / 255.0
		# Normalize with processor stats
		image = (image - image_mean) / image_std
		return image

	def _preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
		image = _load_image(path)
		return image, label

	if shuffle:
		ds = ds.shuffle(buffer_size=min(len(paths), 1000), reshuffle_each_iteration=True)
	
	ds = ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
	ds = ds.batch(batch_size).prefetch(AUTOTUNE)
	return ds


def get_last_conv_or_dense_layer(model: tf.keras.Model) -> str:
	# Heuristic: prefer the last layer named 'classifier' or last dense/conv layer
	for layer in reversed(model.layers):
		name = layer.name.lower()
		if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)) or "classifier" in name or "pooler" in name:
			return layer.name
	return model.layers[-1].name


def save_hf_style(output_dir: str, model: tf.keras.Model, label_map: Dict[str, int], image_size: Tuple[int, int], last_layer_name: str, processor_name: str, id2label: Dict[int, str], label2id: Dict[str, int]) -> None:
	os.makedirs(output_dir, exist_ok=True)
	# Save weights and SavedModel
	weights_path = os.path.join(output_dir, "tf_model.h5")
	model.save_weights(weights_path)
	saved_model_dir = os.path.join(output_dir, "saved_model")
	model.save(saved_model_dir)

	# Save config.json to include important metadata for later XAI
	config = {
		"image_size": [image_size[0], image_size[1], 3],
		"num_labels": int(model.output_shape[-1]),
		"last_layer_name": last_layer_name,
		"processor_name": processor_name,
		"id2label": {str(i): name for i, name in id2label.items()},
		"label2id": label2id
	}
	with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
		json.dump(config, f, indent=2)

	with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
		json.dump({str(v): k for k, v in label_map.items()}, f, indent=2)


def evaluate_and_save_metrics(model: tf.keras.Model, ds: tf.data.Dataset, output_dir: str, id2label: Dict[int, str]):
	# Collect predictions
	y_true: List[int] = []
	y_pred: List[int] = []
	for batch_images, batch_labels in ds:
		logits = model(batch_images, training=False).logits if hasattr(model, 'logits') else model(batch_images, training=False)
		preds = tf.argmax(logits, axis=-1)
		y_true.extend(batch_labels.numpy().tolist())
		y_pred.extend(preds.numpy().tolist())

	acc = float(accuracy_score(y_true, y_pred))
	precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
	cm = confusion_matrix(y_true, y_pred).tolist()

	metrics = {
		"accuracy": acc,
		"precision_weighted": float(precision),
		"recall_weighted": float(recall),
		"f1_weighted": float(f1),
		"confusion_matrix": cm,
		"id2label": {str(k): v for k, v in id2label.items()}
	}
	os.makedirs(output_dir, exist_ok=True)
	with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	# Also save a CSV
	pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


def build_model(model_name: str, num_labels: int, id2label: Dict[int, str], label2id: Dict[str, int]) -> Tuple[tf.keras.Model, AutoImageProcessor]:
	processor = AutoImageProcessor.from_pretrained(model_name)
	# Some ViT checkpoints ship only PyTorch weights; attempt TF init from config if loading fails
	try:
		model = TFAutoModelForImageClassification.from_pretrained(
			model_name,
			num_labels=num_labels,
			id2label=id2label,
			label2id=label2id
		)
	except Exception:
		from transformers import AutoConfig
		config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
		model = TFAutoModelForImageClassification.from_config(config)
	return model, processor


def main():
	parser = argparse.ArgumentParser(description="Fine-tune ViT in TensorFlow and export in HF style")
	parser.add_argument("--data_dir", type=str, required=True, help="Path with subfolders per class")
	parser.add_argument("--output_dir", type=str, required=True, help="Where to save model and artifacts")
	parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="HF model id")
	parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--test_ratio", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	label_map = create_label_map_from_folders(args.data_dir)
	id2label = {i: name for name, i in label_map.items()}
	label2id = {name: i for name, i in label_map.items()}

	all_paths, all_labels = list_image_paths_and_labels(args.data_dir, label_map)
	(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = train_val_test_split(
		all_paths, all_labels, args.val_ratio, args.test_ratio, args.seed
	)

	model, processor = build_model(args.model_name, num_labels=len(label_map), id2label=id2label, label2id=label2id)

	train_ds = build_tf_dataset(train_paths, train_labels, tuple(args.image_size), args.batch_size, True, processor)
	val_ds = build_tf_dataset(val_paths, val_labels, tuple(args.image_size), args.batch_size, False, processor)
	test_ds = build_tf_dataset(test_paths, test_labels, tuple(args.image_size), args.batch_size, False, processor)

	# Define loss/metrics/optimizer
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
	optimizer = keras.optimizers.Adam(learning_rate=args.lr)
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)

	# Callbacks
	ckpt_path = os.path.join(args.output_dir, "checkpoint.keras")
	callbacks = [
		keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True),
		keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
		keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
	]

	model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

	last_layer_name = get_last_conv_or_dense_layer(model)
	save_hf_style(args.output_dir, model, label_map, tuple(args.image_size), last_layer_name, args.model_name, id2label, label2id)

	# Evaluate on test set
	evaluate_and_save_metrics(model, test_ds, args.output_dir, id2label)

	print(f"Training complete. Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
	main()