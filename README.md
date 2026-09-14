# 🍎 Fruit Quality Classification System (FruitNet)
> **Semester Project — Deep Learning & Web Deployment**

---

## 📌 Executive Summary

This project implements an end-to-end Computer Vision system for **Fruit Quality Classification and Freshness Assessment** across 17 distinct fruit categories. Using Deep Transfer Learning (EfficientNetB0 / MobileNetV2 architectures), the system classifies fruits into **Good Quality**, **Bad Quality**, and **Mixed Quality** states.

The project includes:
1. Data Preprocessing & Splitting Pipeline (`01_preprocess.py`)
2. Deep Transfer Learning Model Training Pipeline (`02_train.py`)
3. RESTful Flask Backend API (`app.py`)
4. Modern Interactive Web User Interface (`static/`)

---

## 📁 Semester Project File & Directory Map

All project source files, dataset splits, trained model artifacts, and evaluation reports are stored locally in this project directory:

```text
Fruit/
├── 📄 README.md                            <- Project documentation & report
├── 📄 requirements.txt                     <- Python dependencies
├── 📄 01_preprocess.py                     <- Dataset download, organization & split script
├── 📄 02_train.py                          <- Deep learning model training script
├── 📄 app.py                               <- Flask REST API backend server
├── 📄 preprocessing_output.txt             <- Executed preprocessing report & metrics log
├── 📄 training_output.txt                  <- Executed model training log & evaluation metrics
├── 📄 colab_train_fast.py                  <- High-speed Google Colab GPU training script
├── 📄 FruitNet_Colab_GPU_Fast.ipynb        <- Uploadable Colab Notebook file
│
├── 📂 Fruits10/
│   ├── 📂 metadata/
│   │   └── 📄 metadata.json               <- Class mappings & input dimensions
│   ├── 📂 models/
│   │   ├── 📄 fruit_quality_efficientnetb0.keras <- Trained Keras Model Weights
│   │   ├── 🖼️ training_history.png        <- Loss & Accuracy Training Curves Chart
│   │   └── 🖼️ confusion_matrix.png       <- Evaluation Confusion Matrix Heatmap
│   └── 📂 splits/
│       ├── 📄 train.csv                   <- Training Manifest (70%)
│       ├── 📄 validation.csv              <- Validation Manifest (5%)
│       └── 📄 test.csv                    <- Testing Manifest (25%)
│
└── 📂 static/
    ├── 📄 index.html                      <- Frontend Web UI (Glassmorphism design)
    ├── 📄 style.css                       <- CSS design system & dynamic animations
    └── 📄 app.js                          <- AJAX API connection & camera integration
```

---

## 📊 Preprocessing & Model Evaluation Results

### 1. Preprocessing Summary ([preprocessing_output.txt](file:///c:/Users/PARAGI/Downloads/Banana/Fruit/preprocessing_output.txt))
- **Total Classes**: 17 categories (`apple_Good`, `apple_Bad`, `banana_Good`, `banana_Bad`, `orange_Good`, etc.)
- **Split Ratio**: 70% Training \| 5% Validation \| 25% Testing

### 2. Model Performance Summary ([training_output.txt](file:///c:/Users/PARAGI/Downloads/Banana/Fruit/training_output.txt))
- **Validation Accuracy**: **97.7%**
- **Training Accuracy**: **85.1%**
- **Validation Loss**: **0.5785**
- **Artifacts Saved**:
  - Model Weights: [fruit_quality_efficientnetb0.keras](file:///c:/Users/PARAGI/Downloads/Banana/Fruit/Fruits10/models/fruit_quality_efficientnetb0.keras)
  - Loss/Accuracy Curves: [training_history.png](file:///c:/Users/PARAGI/Downloads/Banana/Fruit/Fruits10/models/training_history.png)
  - Confusion Matrix: [confusion_matrix.png](file:///c:/Users/PARAGI/Downloads/Banana/Fruit/Fruits10/models/confusion_matrix.png)

---

## 🚀 How to Run & Present the Application

### 1. Launch the Server
In your terminal, navigate to the project directory and run:
```bash
python app.py
```

### 2. Open the Web Application
Open your browser and navigate to:
👉 **`http://localhost:5000`**

### 3. Demo Features:
- **Drag & Drop Upload**: Upload any fruit image to classify quality and freshness.
- **Real-Time Camera**: Take live photos directly using your webcam.
- **Interactive Metrics**: Displays top prediction confidence score and class probabilities.

---

## ⚡ Keep-Alive & Self-Ping Mechanism (Prevent 14-Minute Idle Timeout)

Free cloud hosting platforms (like **Render.com**) put web instances to sleep if no HTTP request is received for 15 minutes. To ensure the link remains active 24/7 without turning off after 14 minutes, **FruitNet** implements a dual self-ping system:

1. **Server-Side Background Thread (`app.py`)**:
   - A background daemon thread automatically sends an HTTP request to `/api/health` every **10 minutes** (600s).
   - Automatically detects the deployed public URL via `RENDER_EXTERNAL_URL` or `APP_URL`.

2. **Client-Side Web Heartbeat (`app.js`)**:
   - Sends a lightweight `/api/health` heartbeat every **5 minutes** while the web page is open.

3. **Optional External Monitor (UptimeRobot / Cron-Job)**:
   - For 100% 24/7 zero-downtime uptime, you can paste your Render URL `https://your-app.onrender.com/api/health` into free services like [UptimeRobot.com](https://uptimerobot.com) or [cron-job.org](https://cron-job.org) set to ping every 10 minutes.

