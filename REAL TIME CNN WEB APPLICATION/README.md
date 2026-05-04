# NeuralLens — CNN Image Classifier

A full-stack machine learning web app built with **Flask + TensorFlow/Keras**.

---

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Project Structure

```
my_cnn_app/
├── app.py              # Flask routes & background training thread
├── model_builder.py    # CNN architecture, training loop, inference helpers
├── requirements.txt
├── dataset/            # Auto-created; one sub-folder per class
├── models/             # Saved model.keras + meta.json (auto-created)
├── static/             # confusion_matrix.png (auto-created)
└── templates/
    ├── index.html      # Dataset manager + Predict tab
    └── train.html      # Live training dashboard
```

---

## Workflow

### 1. Create classes & add images — `/ → Dataset tab`
- Click **Add Class** to create a named category.
- Select a class card, then drag-and-drop images into the upload zone.
- Switch to the **Webcam Capture** tab to live-capture frames directly.

### 2. Train — `/train-page`
| Setting | Recommended | Notes |
|---------|-------------|-------|
| Epochs | 20–50 | More = slower but often better |
| Image Size | 150 × 150 | Larger = slower; 224 if you have many images |
| Batch Size | 16 | Reduce to 8 if you get OOM errors |

Click **▶ Start Training** and watch the live accuracy/loss charts update each epoch.

After training the **confusion matrix** appears at the bottom.

### 3. Predict — `/ → Predict tab`
- Upload any image to classify it instantly.
- Click **Use Webcam** for live real-time classification (updates every 1.5 s).

---

## CNN Architecture

```
Input (150×150×3)
  └─ Conv2D 32 → BN → MaxPool → Dropout 0.25
  └─ Conv2D 64 → BN → MaxPool → Dropout 0.25
  └─ Conv2D 128 → BN → MaxPool → Dropout 0.25
  └─ Flatten → Dense 256 (L2) → Dropout 0.5
  └─ Dense num_classes (Softmax)
```

**Data augmentation** (via `ImageDataGenerator`): random rotation, shifts, shear, zoom, horizontal flip — effective for small datasets (30–50 images/class).

**Callbacks**: ReduceLROnPlateau + EarlyStopping with best-weight restoration.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/classes` | List classes + image counts |
| POST | `/api/classes` | Create a class `{name}` |
| DELETE | `/api/classes/<name>` | Delete a class |
| POST | `/api/upload` | Upload images (multipart) |
| POST | `/api/capture` | Save webcam snapshot (base64) |
| POST | `/api/train` | Start training `{epochs, img_size, batch_size}` |
| GET | `/api/train/status` | Poll training progress |
| POST | `/api/predict` | Classify image (file or base64) |

---

## Tips

- **Minimum data**: aim for ≥ 30 images per class before training.
- **Overfitting**: if val_accuracy plateaus while train_accuracy climbs, reduce epochs or add more images.
- **GPU acceleration**: install `tensorflow[and-cuda]` and set up CUDA drivers for 10–50× speedup.
- **Model reuse**: `models/model.keras` persists across restarts; predict works immediately after the server comes back up.
