import os
import shutil
import json
import numpy as np

src_path = r"C:\Users\PARAGI\Downloads\results\fruitnet_18class\best_fruitnet_18class.keras"
dest_dir = r"C:\Users\PARAGI\Downloads\frooit\Fruit\Fruits10\models"
dest_path = os.path.join(dest_dir, "fruit_quality_efficientnetb0.keras")

print(f"Source file exists: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    print(f"Source file size: {size_mb:.2f} MB")
    
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    print(f"Successfully copied to: {dest_path}")
    
    # Try loading with Keras / TensorFlow to check locally
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        model = tf.keras.models.load_model(dest_path)
        print("Model loaded successfully!")
        print(f"Model Input Shape: {model.input_shape}")
        print(f"Model Output Shape: {model.output_shape}")
    except Exception as e:
        print(f"Error checking model: {e}")
else:
    print(f"Error: Source file not found at {src_path}")
