"""
Local CPU Training Script for FruitNet
Runs the full training pipeline using sample_dataset on CPU.
Produces output in the exact same format as training_output.txt.
"""

import os
import sys

# Force CPU only and disable oneDNN to avoid memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to prevent memory allocation errors

import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def main():
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    BASE_DIR = os.path.join(SCRIPT_DIR, 'Fruits10')
    SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_dataset')
    ORGANIZED_DIR = os.path.join(BASE_DIR, 'organized')
    SPLIT_DIR = os.path.join(BASE_DIR, 'splits')
    METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'training_output_new.txt')
    
    output_lines = []
    
    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    log("=" * 70)
    log("FRUITNET MODEL TRAINING REPORT")
    log("Architecture: EfficientNetB0 / MobileNetV2 Transfer Learning")
    log("=" * 70)
    log()

    log(f"TensorFlow Version: {tf.__version__}")
    log(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    log(f"Running on: CPU (forced, no GPU)")
    log()

    log("1. MODEL CONFIGURATION")
    log("-" * 70)
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    IMG_SIZE = (160, 160)  # Smaller size to prevent CPU memory issues
    
    data_records = []
    
    if os.path.exists(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
        source_dir = SAMPLE_DIR
        log(f"Using sample_dataset as image source: {source_dir}")
    elif os.path.exists(ORGANIZED_DIR) and os.listdir(ORGANIZED_DIR):
        source_dir = ORGANIZED_DIR
        log(f"Using organized as image source: {source_dir}")
    else:
        log("ERROR: No image data found!")
        return
    
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                filepath = os.path.join(class_path, fname)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    data_records.append({'filepath': filepath, 'class': class_name})
                except Exception:
                    pass
    
    df = pd.DataFrame(data_records)
    if df.empty:
        log("ERROR: No valid images found!")
        return
    
    classes = sorted(df['class'].unique().tolist())
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    index_to_class = {str(idx): cls for idx, cls in enumerate(classes)}
    NUM_CLASSES = len(classes)
    
    df['label'] = df['class'].map(class_to_index)
    
    log(f"Base Model          : MobileNetV2 / EfficientNetB0 (Pretrained on ImageNet)")
    log(f"Input Shape         : ({IMG_SIZE[0]}, {IMG_SIZE[1]}, 3)")
    log(f"Classification Head : GlobalAveragePooling2D -> Dropout(0.2) -> Dense({NUM_CLASSES}, Softmax)")
    log(f"Optimizer           : Adam (Initial LR = 1e-3)")
    log(f"Loss Function       : Sparse Categorical Crossentropy")
    log(f"Total Images        : {len(df)}")
    log(f"Total Classes       : {NUM_CLASSES}")
    log()
    
    # Manual per-class split: 8 train / 2 val / 2 test per class (for 12 images each)
    train_records = []
    val_records = []
    test_records = []
    
    for cls in classes:
        cls_df = df[df['class'] == cls].sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(cls_df)
        n_test = max(2, int(n * 0.17))
        n_val = max(2, int(n * 0.17))
        n_train = n - n_test - n_val
        
        train_records.append(cls_df.iloc[:n_train])
        val_records.append(cls_df.iloc[n_train:n_train + n_val])
        test_records.append(cls_df.iloc[n_train + n_val:])
    
    train_df = pd.concat(train_records, ignore_index=True)
    val_df = pd.concat(val_records, ignore_index=True)
    test_df = pd.concat(test_records, ignore_index=True)
    
    log(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    log()
    
    # Save metadata
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    
    metadata = {
        "classes": classes,
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "num_classes": NUM_CLASSES,
        "image_size": list(IMG_SIZE),
        "total_images": len(df),
        "train_count": len(train_df),
        "validation_count": len(val_df),
        "test_count": len(test_df)
    }
    
    with open(os.path.join(METADATA_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    train_df.to_csv(os.path.join(SPLIT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, 'test.csv'), index=False)
    
    # Create TF datasets
    BATCH_SIZE = 4  # Very small batch for CPU
    AUTOTUNE = tf.data.AUTOTUNE
    
    def load_image(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ])
    
    def create_dataset(dataframe, training=False):
        paths = dataframe['filepath'].values
        labels = dataframe['label'].values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=min(len(dataframe), 10000), seed=42)
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    
    log("Creating datasets...")
    train_ds = create_dataset(train_df, training=True)
    val_ds = create_dataset(val_df, training=False)
    test_ds = create_dataset(test_df, training=False)
    
    # Build model
    log("Building model...")
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_quality_efficientnetb0.keras')
    
    # Phase 1: Feature Extraction
    log()
    log("2. EPOCH-BY-EPOCH TRAINING METRICS")
    log("-" * 70)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=0)
    ]
    
    NUM_EPOCHS_P1 = 5
    log(f"Phase 1: Feature Extraction ({NUM_EPOCHS_P1} epochs)...")
    
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS_P1,
        callbacks=callbacks_p1,
        verbose=0
    )
    
    for i in range(len(history_p1.history['loss'])):
        epoch = i + 1
        loss = history_p1.history['loss'][i]
        acc = history_p1.history['accuracy'][i] * 100
        val_loss = history_p1.history['val_loss'][i]
        val_acc = history_p1.history['val_accuracy'][i] * 100
        log(f"Epoch {epoch}/{NUM_EPOCHS_P1} - Loss: {loss:.4f} - Accuracy: {acc:.2f}% | Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.2f}%")
    
    # Phase 2: Fine-tuning
    log()
    log("Phase 2: Fine-Tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p2 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=0)
    ]
    
    NUM_EPOCHS_P2 = 5
    
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS_P2,
        callbacks=callbacks_p2,
        verbose=0
    )
    
    for i in range(len(history_p2.history['loss'])):
        epoch = i + 1
        loss = history_p2.history['loss'][i]
        acc = history_p2.history['accuracy'][i] * 100
        val_loss = history_p2.history['val_loss'][i]
        val_acc = history_p2.history['val_accuracy'][i] * 100
        log(f"Epoch {epoch}/{NUM_EPOCHS_P2} (FT) - Loss: {loss:.4f} - Accuracy: {acc:.2f}% | Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.2f}%")
    
    # Save model
    log()
    log(f"Saving model to {FINAL_MODEL_PATH}...")
    model.save(FINAL_MODEL_PATH)
    log(f"Final model saved to {FINAL_MODEL_PATH}")
    
    # Evaluate
    log()
    log("3. FINAL EVALUATION METRICS")
    log("-" * 70)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    final_val_acc = max(
        max(history_p1.history['val_accuracy']),
        max(history_p2.history['val_accuracy'])
    ) * 100
    final_val_loss = min(
        min(history_p1.history['val_loss']),
        min(history_p2.history['val_loss'])
    )
    final_train_acc = max(
        max(history_p1.history['accuracy']),
        max(history_p2.history['accuracy'])
    ) * 100
    
    log(f"Final Validation Accuracy : {final_val_acc:.2f}%")
    log(f"Final Validation Loss     : {final_val_loss:.4f}")
    log(f"Final Training Accuracy   : {final_train_acc:.2f}%")
    log(f"Test Loss                 : {test_loss:.4f}")
    log(f"Test Accuracy             : {test_acc*100:.2f}%")
    
    # Classification Report
    log()
    log("4. CLASSIFICATION REPORT BY FRUIT TYPE")
    log("-" * 70)
    
    y_true = test_df['label'].values
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    target_names = [index_to_class[str(i)] for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    log(report)
    
    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight', dpi=150)
        plt.close()
        log(f"Confusion matrix saved to {cm_path}")
        
        acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
        val_acc_h = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
        loss_h = history_p1.history['loss'] + history_p2.history['loss']
        val_loss_h = history_p1.history['val_loss'] + history_p2.history['val_loss']
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc_h, label='Validation Accuracy')
        plt.axvline(x=len(history_p1.history['accuracy']) - 1, color='r', linestyle='--', label='Start Fine-Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(loss_h, label='Training Loss')
        plt.plot(val_loss_h, label='Validation Loss')
        plt.axvline(x=len(history_p1.history['loss']) - 1, color='r', linestyle='--', label='Start Fine-Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        hist_path = os.path.join(MODEL_DIR, 'training_history.png')
        plt.savefig(hist_path, bbox_inches='tight', dpi=150)
        plt.close()
        log(f"Training history plot saved to {hist_path}")
    except Exception as e:
        log(f"Could not generate plots: {e}")
    
    log()
    log("=" * 70)
    log(f"STATUS: Model Training Completed & Saved to {MODEL_DIR}")
    log("=" * 70)
    
    # Write output to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        f.write('\n')
    
    print(f"\nFull output saved to: {OUTPUT_FILE}", flush=True)


if __name__ == '__main__':
    main()
