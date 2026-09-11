import os
import sys
import shutil
import zipfile
import subprocess
import glob
from pathlib import Path
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Setup directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Fruits10'))
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fruitnet'))
EXTRACT_DIR = os.path.join(RAW_DIR, 'extracted')
ORGANIZED_DIR = os.path.join(BASE_DIR, 'organized')
SPLIT_DIR = os.path.join(BASE_DIR, 'splits')
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create necessary directories
for d in [BASE_DIR, RAW_DIR, EXTRACT_DIR, ORGANIZED_DIR, SPLIT_DIR, METADATA_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_NAME = 'shashwatwork/fruitnet-indian-fruits-dataset-with-quality'

MAPPING = {
    os.path.join('Bad Quality_Fruits', 'Apple_Bad'): 'apple_Bad',
    os.path.join('Bad Quality_Fruits', 'Banana_Bad'): 'banana_Bad',
    os.path.join('Bad Quality_Fruits', 'Guava_Bad'): 'guava_Bad',
    os.path.join('Bad Quality_Fruits', 'Lime_Bad'): 'lime_Bad',
    os.path.join('Bad Quality_Fruits', 'Orange_Bad'): 'orange_Bad',
    os.path.join('Bad Quality_Fruits', 'Pomegranate_Bad'): 'pomegranate_Bad',
    
    os.path.join('Good Quality_Fruits', 'Apple_Good'): 'apple_Good',
    os.path.join('Good Quality_Fruits', 'Banana_Good'): 'banana_Good',
    os.path.join('Good Quality_Fruits', 'Guava_Good'): 'guava_Good',
    os.path.join('Good Quality_Fruits', 'Lime_Good'): 'lime_Good',
    os.path.join('Good Quality_Fruits', 'Orange_Good'): 'orange_Good',
    os.path.join('Good Quality_Fruits', 'Pomegranate_Good'): 'pomegranate_Good',
    
    os.path.join('Mixed Qualit_Fruits', 'Apple'): 'apple_Mixed',
    os.path.join('Mixed Qualit_Fruits', 'Banana'): 'banana_Mixed',
    os.path.join('Mixed Qualit_Fruits', 'Guava'): 'guava_Mixed',
    os.path.join('Mixed Qualit_Fruits', 'Orange'): 'orange_Mixed',
    os.path.join('Mixed Qualit_Fruits', 'Pomegranate'): 'pomegranate_Mixed',
    os.path.join('Mixed Qualit_Fruits', 'Lemon'): 'lime_Mixed',
}

def setup_kaggle():
    print("Setting up Kaggle credentials...")
    kaggle_json_path = os.path.join(os.path.dirname(__file__), 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print(f"Warning: {kaggle_json_path} not found. Kaggle CLI might fail if not already configured.")
        return

    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    dest_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(dest_path):
        shutil.copy(kaggle_json_path, dest_path)
        # Try to restrict permissions if on a UNIX-like system (not strictly needed for Windows but good practice)
        try:
            os.chmod(dest_path, 0o600)
        except:
            pass
    print("Kaggle credentials setup complete.")

def download_and_extract():
    zip_path = os.path.join(RAW_DIR, 'fruitnet-indian-fruits-dataset-with-quality.zip')
    
    if not os.path.exists(zip_path) and not os.path.exists(os.path.join(EXTRACT_DIR, 'Processed Images_Fruits')):
        print("Downloading dataset using Kaggle CLI...")
        try:
            subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', DATASET_NAME, '-p', RAW_DIR],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("Error: Kaggle CLI not found. Please pip install kaggle.")
            sys.exit(1)
    else:
        print("Dataset zip already exists or already extracted, skipping download.")

    if not os.listdir(EXTRACT_DIR):
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {EXTRACT_DIR}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
            print("Extraction complete.")
        else:
            print(f"Could not find {zip_path} to extract.")
    else:
        print("Extraction directory not empty, skipping extraction.")

def walk_tree(d, level=0):
    if not os.path.isdir(d):
        return
    indent = '  ' * level
    print(f"{indent}{os.path.basename(d)}/")
    for item in os.listdir(d):
        item_path = os.path.join(d, item)
        if os.path.isdir(item_path):
            walk_tree(item_path, level + 1)

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def organize_images():
    print("\nOrganizing images...")
    processed_dir = os.path.join(EXTRACT_DIR, 'Processed Images_Fruits')
    if not os.path.exists(processed_dir):
        # Fallback if structure is slightly different
        processed_dir = EXTRACT_DIR
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_found = 0
    total_valid = 0
    total_corrupt = 0
    
    data = []

    for root, dirs, files in os.walk(processed_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                total_found += 1
                src_path = os.path.join(root, f)
                
                # Determine which mapping bucket it falls into based on path
                rel_path = os.path.relpath(root, processed_dir)
                
                # Match rel_path to mapping
                matched_class = None
                for map_key, class_name in MAPPING.items():
                    if rel_path.endswith(map_key) or map_key in rel_path:
                        matched_class = class_name
                        break
                
                if matched_class is None:
                    continue
                
                # Validate image
                if not is_valid_image(src_path):
                    print(f"Warning: Skipping corrupt image {src_path}")
                    total_corrupt += 1
                    continue
                
                class_dir = os.path.join(ORGANIZED_DIR, matched_class)
                os.makedirs(class_dir, exist_ok=True)
                
                # Ensure unique filename
                dest_path = os.path.join(class_dir, f)
                idx = 1
                while os.path.exists(dest_path):
                    name, e = os.path.splitext(f)
                    dest_path = os.path.join(class_dir, f"{name}_{idx}{e}")
                    idx += 1
                
                shutil.copy2(src_path, dest_path)
                data.append({'filepath': dest_path, 'class': matched_class})
                total_valid += 1

                if total_valid % 1000 == 0:
                    print(f"Organized {total_valid} images...")

    print(f"Finished organizing. Found: {total_found}, Valid: {total_valid}, Corrupt: {total_corrupt}")
    return pd.DataFrame(data)

def main():
    setup_kaggle()
    download_and_extract()
    
    print("\nDirectory Structure after extraction:")
    walk_tree(EXTRACT_DIR, level=1)
    
    df = organize_images()
    if df.empty:
        print("No images found or organized! Exiting.")
        return
        
    print(f"\nCreated DataFrame with {len(df)} images.")
    
    # Sort classes, mappings
    classes = sorted(df['class'].unique().tolist())
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    index_to_class = {str(idx): cls for idx, cls in enumerate(classes)}
    
    df['label'] = df['class'].map(class_to_index)
    
    # Splits (70 / 5 / 25)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    # Remaining 30%: split into ~5% val and ~25% test -> test_size=25/30=5/6
    val_df, test_df = train_test_split(temp_df, test_size=(25.0/30.0), stratify=temp_df['label'], random_state=42)
    
    train_csv = os.path.join(SPLIT_DIR, 'train.csv')
    val_csv = os.path.join(SPLIT_DIR, 'validation.csv')
    test_csv = os.path.join(SPLIT_DIR, 'test.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved splits to {SPLIT_DIR}")
    
    # Metadata
    metadata = {
        "classes": classes,
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "num_classes": len(classes),
        "image_size": [224, 224],
        "total_images": len(df),
        "train_count": len(train_df),
        "validation_count": len(val_df),
        "test_count": len(test_df)
    }
    
    meta_path = os.path.join(METADATA_DIR, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {meta_path}")
    
    print("\n=== Summary Statistics ===")
    print(f"Total Images Organized: {metadata['total_images']}")
    print(f"Classes (17 expected): {metadata['num_classes']}")
    print(f"Train split: {metadata['train_count']}")
    print(f"Validation split: {metadata['validation_count']}")
    print(f"Test split: {metadata['test_count']}")
    
if __name__ == '__main__':
    main()
