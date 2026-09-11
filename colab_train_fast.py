"""
=============================================================
FruitNet - FAST Combined Preprocessing + Training for Colab
=============================================================
INSTRUCTIONS:
1. Open Google Colab: https://colab.research.google.com
2. Go to Runtime > Change runtime type > Select GPU (T4)
3. Upload this file or paste all code into a cell
4. Run it - takes ~15-20 minutes
5. It will auto-download model + metadata at the end
=============================================================
"""

# ===================== CELL 1: Install dependencies =====================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle", "scikit-learn", "seaborn"])

# ===================== CELL 2: Imports =====================
import os
import json
import shutil
import zipfile
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import seaborn as sns

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ===================== CELL 3: Setup Kaggle =====================
from google.colab import files

# Upload kaggle.json
print("Please upload your kaggle.json file:")
uploaded = files.upload()

os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Kaggle API configured.")

# ===================== CELL 4: Configuration =====================
BASE_DIR = "/content/Fruits10"
RAW_DIR = "/content/fruitnet"
EXTRACT_DIR = "/content/fruitnet/extracted"
ORGANIZED_DIR = f"{BASE_DIR}/organized"
SPLIT_DIR = f"{BASE_DIR}/splits"
METADATA_DIR = f"{BASE_DIR}/metadata"
MODEL_DIR = f"{BASE_DIR}/models"

for folder in [RAW_DIR, EXTRACT_DIR, ORGANIZED_DIR, SPLIT_DIR, METADATA_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

print("Directories created at:", BASE_DIR)

# ===================== CELL 5: Download Dataset =====================
print("\n=== STEP 1/6: Downloading dataset (~3 GB)... ===")
os.system(f"kaggle datasets download -d shashwatwork/fruitnet-indian-fruits-dataset-with-quality -p {RAW_DIR}")

# ===================== CELL 6: Extract =====================
print("\n=== STEP 2/6: Extracting... ===")
zip_files = glob.glob(f"{RAW_DIR}/*.zip")
if zip_files:
    with zipfile.ZipFile(zip_files[0], "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Extraction complete.")

# ===================== CELL 7: Organize Images =====================
print("\n=== STEP 3/6: Organizing images into classes... ===")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

MAPPING = {
    "Bad Quality_Fruits/Apple_Bad": "apple_Bad",
    "Bad Quality_Fruits/Banana_Bad": "banana_Bad",
    "Bad Quality_Fruits/Guava_Bad": "guava_Bad",
    "Bad Quality_Fruits/Lime_Bad": "lime_Bad",
    "Bad Quality_Fruits/Orange_Bad": "orange_Bad",
    "Bad Quality_Fruits/Pomegranate_Bad": "pomegranate_Bad",
    "Good Quality_Fruits/Apple_Good": "apple_Good",
    "Good Quality_Fruits/Banana_Good": "banana_Good",
    "Good Quality_Fruits/Guava_Good": "guava_Good",
    "Good Quality_Fruits/Lime_Good": "lime_Good",
    "Good Quality_Fruits/Orange_Good": "orange_Good",
    "Good Quality_Fruits/Pomegranate_Good": "pomegranate_Good",
    "Mixed Qualit_Fruits/Apple": "apple_Mixed",
    "Mixed Qualit_Fruits/Banana": "banana_Mixed",
    "Mixed Qualit_Fruits/Guava": "guava_Mixed",
    "Mixed Qualit_Fruits/Orange": "orange_Mixed",
    "Mixed Qualit_Fruits/Pomegranate": "pomegranate_Mixed",
    "Mixed Qualit_Fruits/Lemon": "lime_Mixed",
}

processed_dir = f"{EXTRACT_DIR}/Processed Images_Fruits"
data_records = []
count = 0

for root, dirs, files_list in os.walk(processed_dir):
    for fname in files_list:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        src_path = os.path.join(root, fname)
        rel_path = os.path.relpath(root, processed_dir)

        matched_class = None
        for map_key, class_name in MAPPING.items():
            if map_key in rel_path or rel_path.endswith(map_key):
                matched_class = class_name
                break

        if matched_class is None:
            continue

        # Validate image
        try:
            with Image.open(src_path) as img:
                img.verify()
        except Exception:
            continue

        class_dir = os.path.join(ORGANIZED_DIR, matched_class)
        os.makedirs(class_dir, exist_ok=True)

        dest_path = os.path.join(class_dir, fname)
        idx = 1
        while os.path.exists(dest_path):
            name, e = os.path.splitext(fname)
            dest_path = os.path.join(class_dir, f"{name}_{idx}{e}")
            idx += 1

        shutil.copy2(src_path, dest_path)
        data_records.append({"filepath": dest_path, "class": matched_class})
        count += 1

        if count % 2000 == 0:
            print(f"  Organized {count} images...")

df = pd.DataFrame(data_records)
print(f"Total images organized: {len(df)}")

# ===================== CELL 8: Create Splits + Metadata =====================
print("\n=== STEP 4/6: Creating train/val/test splits... ===")

classes = sorted(df["class"].unique().tolist())
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
index_to_class = {str(idx): cls for idx, cls in enumerate(classes)}
NUM_CLASSES = len(classes)

df["label"] = df["class"].map(class_to_index)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=(25.0/30.0), stratify=temp_df["label"], random_state=42)

train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
val_df.to_csv(f"{SPLIT_DIR}/validation.csv", index=False)
test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

IMG_SIZE = (224, 224)

metadata = {
    "classes": classes,
    "class_to_index": class_to_index,
    "index_to_class": index_to_class,
    "num_classes": NUM_CLASSES,
    "image_size": list(IMG_SIZE),
    "total_images": len(df),
    "train_count": len(train_df),
    "validation_count": len(val_df),
    "test_count": len(test_df),
}

with open(f"{METADATA_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Classes: {NUM_CLASSES}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ===================== CELL 9: Build Datasets =====================
print("\n=== STEP 5/6: Training model (GPU accelerated)... ===")

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.10),
])

def create_dataset(dataframe, training=False):
    paths = dataframe["filepath"].values
    labels = dataframe["label"].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=min(len(dataframe), 10000), seed=42)
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    if training:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df, training=False)
test_ds = create_dataset(test_df, training=False)

# ===================== CELL 10: Build + Train Model =====================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.20)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

BEST_MODEL_PATH = f"{MODEL_DIR}/best_fruit_model.keras"
FINAL_MODEL_PATH = f"{MODEL_DIR}/fruit_quality_efficientnetb0.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
    ),
]

# Phase 1: Feature extraction
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Phase 1: Feature Extraction (10 epochs) ---")
history_p1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# Phase 2: Fine-tuning
print("\n--- Phase 2: Fine-Tuning (15 epochs) ---")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_p2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

# ===================== CELL 11: Save + Evaluate =====================
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save(FINAL_MODEL_PATH)
print(f"\nFinal model saved: {FINAL_MODEL_PATH}")

test_loss, test_accuracy = best_model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# ===================== CELL 12: Download Files =====================
print("\n=== STEP 6/6: Downloading model files... ===")
print("Two files will be downloaded:")
print("  1. fruit_quality_efficientnetb0.keras (the trained model)")
print("  2. metadata.json (class mappings)")
print()

from google.colab import files

# Download model
files.download(FINAL_MODEL_PATH)

# Download metadata
files.download(f"{METADATA_DIR}/metadata.json")

print("\n" + "=" * 60)
print("DONE! Now on your local machine:")
print("  1. Create folder: Fruits10/models/ and Fruits10/metadata/")
print("  2. Put fruit_quality_efficientnetb0.keras in Fruits10/models/")
print("  3. Put metadata.json in Fruits10/metadata/")
print("  4. Run: python app.py")
print("  5. Open: http://localhost:5000")
print("=" * 60)
