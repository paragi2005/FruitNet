import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json
import numpy as np
import io
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

print("Loading TensorFlow and Keras... This may take a moment.")
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("Error: tensorflow is not installed. Model will not load.")
    load_model = None

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = './Fruits10/models/fruit_quality_efficientnetb0.keras'
METADATA_PATH = './Fruits10/metadata/metadata.json'

model = None
metadata = {}
classes = []

# Load Metadata
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            classes = metadata.get('classes', [])
    except Exception as e:
        print(f"Error loading metadata: {e}")
else:
    print(f"Warning: Metadata not found at {METADATA_PATH}")

# Load Model
if load_model and os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH} or TF not installed.")

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_class_name(class_name):
    parts = class_name.split('_')
    if len(parts) >= 2:
        fruit = parts[0].capitalize()
        quality = parts[1].capitalize()
        return fruit, quality
    return class_name.capitalize(), "Unknown"

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'], strict_slashes=False)
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": classes
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'], strict_slashes=False)
def predict():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    if model is None:
        return jsonify({"success": False, "error": "Model is loading or not available."}), 503

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image field in request."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Determine model input dimensions safely
        target_size = (160, 160)
        try:
            if model is not None:
                shape = model.input_shape
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                    target_size = (int(shape[2]), int(shape[1])) # (width, height) for PIL
            elif 'image_size' in metadata:
                size_arr = metadata['image_size']
                target_size = (int(size_arr[1]), int(size_arr[0]))
        except Exception as se:
            print(f"Size detection warning: {se}")
            target_size = (160, 160)
            
        print(f"Resizing input image to PIL dimensions (W, H): {target_size}...")
        image = image.resize(target_size)
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        print(f"Raw prediction vector: {preds}")
        
        top_indices = np.argsort(preds)[::-1][:5]
        
        predictions = []
        for i in top_indices:
            class_name = classes[i] if i < len(classes) else f"class_{i}"
            confidence = float(preds[i])
            fruit, quality = parse_class_name(class_name)
            predictions.append({
                "class": class_name,
                "fruit": fruit,
                "quality": quality,
                "confidence": confidence
            })
            
        print(f"Top prediction: {predictions[0] if predictions else None}")
        return jsonify({
            "success": True,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        })

    except Exception as e:
        print(f"Prediction Exception: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Fruit Quality Classification API")
    print("Server running at: http://localhost:5000/")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
