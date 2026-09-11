import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    BASE_DIR = os.path.join(".", "Fruits10")
    SPLIT_DIR = os.path.join(BASE_DIR, "splits")
    METADATA_DIR = os.path.join(BASE_DIR, "metadata")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(METADATA_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please run 01_preprocess.py first.")
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    IMG_SIZE = metadata["image_size"]
    NUM_CLASSES = metadata["num_classes"]
    INDEX_TO_CLASS = metadata["index_to_class"]

    # Load datasets
    train_csv = os.path.join(SPLIT_DIR, "train.csv")
    val_csv = os.path.join(SPLIT_DIR, "validation.csv")
    test_csv = os.path.join(SPLIT_DIR, "test.csv")

    if not all(map(os.path.exists, [train_csv, val_csv, test_csv])):
        print("Error: Missing CSV split files. Please run 01_preprocess.py first.")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Loaded train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Verify paths
    missing_count = sum(1 for p in pd.concat([train_df['filepath'], val_df['filepath'], test_df['filepath']]) if not os.path.exists(p))
    if missing_count > 0:
        print(f"Warning: {missing_count} image paths do not exist!")

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
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10)
    ])

    def create_dataset(dataframe, training=False):
        paths = dataframe['filepath'].values
        labels = dataframe['label'].values
        
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        
        if training:
            buffer_size = min(len(dataframe), 10000)
            ds = ds.shuffle(buffer_size=buffer_size, seed=42)
            
        ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        
        if training:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    print("Creating datasets...")
    train_ds = create_dataset(train_df, training=True)
    val_ds = create_dataset(val_df, training=False)
    test_ds = create_dataset(test_df, training=False)

    print("Building model...")
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

    print("Compiling model for Phase 1...")
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_fruit_model.keras")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "fruit_quality_efficientnetb0.keras")

    callbacks_p1 = [
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    print("Starting Phase 1 (Feature Extraction)...")
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks_p1
    )

    print("Starting Phase 2 (Fine-tuning)...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
        
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks_p2 = [
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks_p2
    )

    print(f"Loading best model from {BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print("Generating predictions for classification report...")
    y_true = test_df['label'].values
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = [INDEX_TO_CLASS[str(i)] for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Generating plots...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # Accuracy Plot
    acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
    val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
    loss = history_p1.history['loss'] + history_p2.history['loss']
    val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history_p1.history['accuracy']) - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=len(history_p1.history['loss']) - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    hist_path = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()

    print(f"Training complete. Models and plots saved to {MODEL_DIR}")

if __name__ == '__main__':
    main()
