import os
import tensorflow as tf

# Patch Dense.__init__ in tensorflow.keras.layers and keras.src.layers.core.dense
from tensorflow.keras.layers import Dense

orig_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    orig_dense_init(self, *args, **kwargs)

Dense.__init__ = patched_dense_init

# Also check standalone keras if present
try:
    import keras
    keras.layers.Dense.__init__ = patched_dense_init
    print("Patched standalone keras.layers.Dense.__init__")
except Exception:
    pass

model_path = r"C:\Users\PARAGI\Downloads\frooit\Fruit\Fruits10\models\fruit_quality_efficientnetb0.keras"

print(f"Loading model from {model_path}...")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("SUCCESS: Model loaded successfully with patched Dense!")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
except Exception as e:
    import traceback
    print("FAILED to load model:")
    traceback.print_exc()
