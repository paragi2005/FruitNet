import os
import tensorflow as tf

model_path = r"C:\Users\PARAGI\Downloads\frooit\Fruit\Fruits10\models\fruit_quality_efficientnetb0.keras"

print("Attempting load_model with compile=False...")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("SUCCESS: Loaded model with compile=False!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print(f"Failed with compile=False: {e}")

print("\nAttempting with custom Dense class / config patch if needed...")
try:
    # Custom Dense layer override to ignore 'quantization_config' argument in older/different Keras versions
    from tensorflow.keras.layers import Dense as KerasDense

    class PatchedDense(KerasDense):
        def __init__(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config.pop('quantization_config', None)
            return super().from_config(config)

    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'Dense': PatchedDense}, 
        compile=False
    )
    print("SUCCESS: Loaded model with PatchedDense!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print(f"Failed with PatchedDense: {e}")
