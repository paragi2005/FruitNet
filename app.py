import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
import io
import threading
import time
import urllib.request

from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ============================================================
# SELF PING - RENDER
# ============================================================

def start_self_ping():

    def ping_worker():

        time.sleep(20)

        while True:

            try:

                app_url = (
                    os.environ.get("RENDER_EXTERNAL_URL")
                    or os.environ.get("APP_URL")
                )

                if app_url:
                    health_url = app_url.rstrip("/") + "/api/health"
                else:
                    health_url = "http://127.0.0.1:5000/api/health"

                req = urllib.request.Request(
                    health_url,
                    headers={
                        "User-Agent": "FruitNet-SelfPing/1.0"
                    }
                )

                with urllib.request.urlopen(
                    req,
                    timeout=10
                ) as resp:

                    print(
                        f"[Self-Ping] {health_url} "
                        f"-> Status: {resp.status} OK"
                    )

            except Exception as e:

                print(f"[Self-Ping] Notice: {e}")

            time.sleep(600)

    thread = threading.Thread(
        target=ping_worker,
        daemon=True
    )

    thread.start()


# ============================================================
# START SELF PING
# ============================================================

start_self_ping()


# ============================================================
# LOAD TENSORFLOW / KERAS
# ============================================================

print("Loading TensorFlow and Keras...")

try:

    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Dense

    original_dense_init = Dense.__init__

    def patched_dense_init(self, *args, **kwargs):

        kwargs.pop("quantization_config", None)

        original_dense_init(
            self,
            *args,
            **kwargs
        )

    Dense.__init__ = patched_dense_init

    try:

        import keras

        keras.layers.Dense.__init__ = (
            patched_dense_init
        )

    except Exception:
        pass

except ImportError:

    print(
        "ERROR: TensorFlow is not installed."
    )

    load_model = None


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static"
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# IMPORTANT:
# USE YOUR NEW 18-CLASS MODEL
# ============================================================

MODEL_PATH = (
    "./Fruits10/models/"
    "best_fruitnet_18class.keras"
)

CLASS_NAMES_PATH = (
    "./Fruits10/metadata/"
    "class_names.json"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
classes = []


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            class_data = json.load(f)

        # Case 1:
        # ["apple_Bad", "apple_Good", ...]
        if isinstance(class_data, list):

            classes = class_data

        # Case 2:
        # {"classes": [...]}
        elif isinstance(class_data, dict):

            if "classes" in class_data:

                classes = class_data["classes"]

            elif "class_names" in class_data:

                classes = class_data["class_names"]

            else:

                # Case 3:
                # {"0": "apple_Bad", "1": "apple_Good", ...}

                try:

                    classes = [
                        class_data[str(i)]
                        for i in range(len(class_data))
                    ]

                except Exception:

                    classes = []

        print()
        print("=" * 60)
        print("CLASS NAMES")
        print("=" * 60)

        print(
            "Number of classes:",
            len(classes)
        )

        for i, name in enumerate(classes):

            print(
                f"{i}: {name}"
            )

        print("=" * 60)
        print()

    except Exception as e:

        print(
            "ERROR loading class_names.json:",
            e
        )

else:

    print()
    print("=" * 60)
    print("ERROR: CLASS NAMES FILE NOT FOUND")
    print(CLASS_NAMES_PATH)
    print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

if (
    load_model is not None
    and os.path.exists(MODEL_PATH)
):

    try:

        print()
        print(
            "Loading model from:"
        )
        print(MODEL_PATH)

        model = load_model(
            MODEL_PATH,
            compile=False
        )

        print(
            "Model loaded successfully."
        )

        print(
            "Model input shape:",
            model.input_shape
        )

        print(
            "Model output shape:",
            model.output_shape
        )

    except Exception as e:

        print(
            "ERROR loading model:",
            e
        )

else:

    print()
    print("=" * 60)
    print("MODEL NOT FOUND")
    print(MODEL_PATH)
    print("=" * 60)


# ============================================================
# VERIFY MODEL OUTPUTS
# ============================================================

if model is not None:

    try:

        output_shape = model.output_shape

        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        model_classes = int(
            output_shape[-1]
        )

        print()
        print(
            "Model output classes:",
            model_classes
        )

        print(
            "Class names available:",
            len(classes)
        )

        if model_classes != len(classes):

            print()
            print("=" * 60)
            print("WARNING: CLASS COUNT MISMATCH")
            print(
                f"Model = {model_classes}"
            )
            print(
                f"Classes = {len(classes)}"
            )
            print("=" * 60)

    except Exception as e:

        print(
            "Class verification error:",
            e
        )


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PARSE CLASS NAME
# ============================================================

def parse_class_name(class_name):

    parts = class_name.split("_")

    if len(parts) >= 2:

        fruit = parts[0].capitalize()
        quality = parts[1].capitalize()

        return fruit, quality

    return (
        class_name.capitalize(),
        "Unknown"
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        app.static_folder,
        "index.html"
    )


# ============================================================
# SERVICE WORKER
# ============================================================

@app.route("/service-worker.js")
def service_worker():

    return send_from_directory(
        app.static_folder,
        "service-worker.js",
        mimetype="application/javascript"
    )


# ============================================================
# MANIFEST
# ============================================================

@app.route("/manifest.json")
def manifest():

    return send_from_directory(
        app.static_folder,
        "manifest.json",
        mimetype="application/json"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"],
    strict_slashes=False
)
def health():

    return jsonify({

        "status": "ok",

        "model_loaded":
            model is not None,

        "model_path":
            MODEL_PATH,

        "number_of_classes":
            len(classes),

        "classes":
            classes
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST", "OPTIONS"],
    strict_slashes=False
)
def predict():

    # CORS preflight
    if request.method == "OPTIONS":

        return jsonify({
            "status": "ok"
        }), 200

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if model is None:

        return jsonify({

            "success": False,

            "error":
                "Model is not available."
        }), 503

    # --------------------------------------------------------
    # IMAGE CHECK
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image field in request."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "Empty filename."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({

            "success": False,

            "error":
                "Invalid file type. Allowed: "
                + ", ".join(
                    sorted(ALLOWED_EXTENSIONS)
                )
        }), 400

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        print()
        print("=" * 60)
        print("NEW PREDICTION")
        print("=" * 60)

        print(
            "File:",
            file.filename
        )

        print(
            "Original size:",
            image.size
        )

        # ----------------------------------------------------
        # GET MODEL INPUT SIZE
        # ----------------------------------------------------

        shape = model.input_shape

        if isinstance(shape, list):

            shape = shape[0]

        if (
            len(shape) != 4
            or shape[1] is None
            or shape[2] is None
        ):

            return jsonify({

                "success": False,

                "error":
                    f"Unexpected model input shape: {shape}"

            }), 500

        height = int(shape[1])
        width = int(shape[2])

        print(
            "Model input:",
            width,
            "x",
            height
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # DO NOT DIVIDE BY 255
        #
        # Your EfficientNetB0 model contains the
        # preprocessing/rescaling expected by the model.
        # ----------------------------------------------------

        img_array = np.array(
            image,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        print(
            "Input shape:",
            img_array.shape
        )

        print(
            "Input minimum:",
            float(img_array.min())
        )

        print(
            "Input maximum:",
            float(img_array.max())
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        preds = model.predict(
            img_array,
            verbose=0
        )[0]

        # ----------------------------------------------------
        # TOP 5
        # ----------------------------------------------------

        top_indices = np.argsort(
            preds
        )[::-1][:5]

        predictions = []

        for i in top_indices:

            i = int(i)

            if i < len(classes):

                class_name = classes[i]

            else:

                class_name = f"class_{i}"

            confidence = float(
                preds[i]
            )

            fruit, quality = (
                parse_class_name(
                    class_name
                )
            )

            predictions.append({

                "class":
                    class_name,

                "fruit":
                    fruit,

                "quality":
                    quality,

                "confidence":
                    confidence,

                "confidence_percent":
                    round(
                        confidence * 100,
                        2
                    )
            })

        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print()
        print("TOP 5 PREDICTIONS")
        print("-" * 60)

        for p in predictions:

            print(
                f"{p['class']:<25}"
                f"{p['confidence_percent']:>8.2f}%"
            )

        print("=" * 60)

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "predictions":
                predictions,

            "top_prediction":
                predictions[0]
                if predictions
                else None
        })

    except Exception as e:

        print(
            "Prediction Exception:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "Starting FruitNet 18-Class "
        "Fruit + Quality Classification API"
    )
    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Number of classes:",
        len(classes)
    )

    print(
        "Server:"
        " http://localhost:5000/"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )