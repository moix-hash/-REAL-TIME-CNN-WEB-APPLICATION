import os
import io
import base64
import threading
import time
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['MODEL_FOLDER'] = 'models'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Global training state
training_state = {
    'status': 'idle',       # idle | training | done | error
    'epoch': 0,
    'total_epochs': 0,
    'accuracy': 0.0,
    'val_accuracy': 0.0,
    'loss': 0.0,
    'val_loss': 0.0,
    'logs': [],
    'classes': [],
    'confusion_matrix_path': None,
    'error': None,
}
training_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Lazy imports (TF is heavy – only pulled in when needed)
# ---------------------------------------------------------------------------
def get_tf():
    import tensorflow as tf
    return tf

def get_model_builder():
    from model_builder import train_model
    return train_model

# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train-page')
def train_page():
    return render_template('train.html')

# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------
@app.route('/api/classes', methods=['GET'])
def list_classes():
    dataset_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(dataset_dir):
        return jsonify({'classes': []})
    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    class_info = []
    for cls in sorted(classes):
        cls_path = os.path.join(dataset_dir, cls)
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        class_info.append({'name': cls, 'count': len(images)})
    return jsonify({'classes': class_info})


@app.route('/api/classes', methods=['POST'])
def create_class():
    data = request.get_json()
    class_name = data.get('name', '').strip()
    if not class_name:
        return jsonify({'error': 'Class name required'}), 400
    safe_name = secure_filename(class_name)
    class_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    os.makedirs(class_path, exist_ok=True)
    return jsonify({'success': True, 'name': safe_name})


@app.route('/api/classes/<class_name>', methods=['DELETE'])
def delete_class(class_name):
    import shutil
    class_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(class_name))
    if os.path.exists(class_path):
        shutil.rmtree(class_path)
    return jsonify({'success': True})


@app.route('/api/upload', methods=['POST'])
def upload_images():
    class_name = request.form.get('class_name', '').strip()
    if not class_name:
        return jsonify({'error': 'class_name required'}), 400
    safe_name = secure_filename(class_name)
    class_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    os.makedirs(class_path, exist_ok=True)

    files = request.files.getlist('images')
    saved = 0
    for f in files:
        if f and f.filename:
            fname = secure_filename(f.filename)
            # avoid collisions
            base, ext = os.path.splitext(fname)
            dest = os.path.join(class_path, fname)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(class_path, f"{base}_{counter}{ext}")
                counter += 1
            f.save(dest)
            saved += 1
    return jsonify({'success': True, 'saved': saved})


@app.route('/api/capture', methods=['POST'])
def capture_image():
    """Accept a base64 webcam snapshot and save it to a class folder."""
    data = request.get_json()
    class_name = data.get('class_name', '').strip()
    image_data = data.get('image', '')
    if not class_name or not image_data:
        return jsonify({'error': 'class_name and image required'}), 400

    safe_name = secure_filename(class_name)
    class_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    os.makedirs(class_path, exist_ok=True)

    # Strip data URL header
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    img_bytes = base64.b64decode(image_data)
    fname = f"capture_{int(time.time() * 1000)}.jpg"
    with open(os.path.join(class_path, fname), 'wb') as fh:
        fh.write(img_bytes)
    return jsonify({'success': True, 'filename': fname})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.route('/api/train', methods=['POST'])
def start_training():
    global training_state
    with training_lock:
        if training_state['status'] == 'training':
            return jsonify({'error': 'Training already in progress'}), 409

    data = request.get_json() or {}
    epochs = int(data.get('epochs', 20))
    img_size = int(data.get('img_size', 150))
    batch_size = int(data.get('batch_size', 16))

    def run():
        global training_state
        with training_lock:
            training_state.update({
                'status': 'training',
                'epoch': 0,
                'total_epochs': epochs,
                'accuracy': 0.0,
                'val_accuracy': 0.0,
                'loss': 0.0,
                'val_loss': 0.0,
                'logs': [],
                'error': None,
                'confusion_matrix_path': None,
            })
        try:
            train_fn = get_model_builder()
            train_fn(
                dataset_dir=app.config['UPLOAD_FOLDER'],
                model_dir=app.config['MODEL_FOLDER'],
                static_dir=app.config['STATIC_FOLDER'],
                epochs=epochs,
                img_size=img_size,
                batch_size=batch_size,
                state=training_state,
                lock=training_lock,
            )
        except Exception as e:
            with training_lock:
                training_state['status'] = 'error'
                training_state['error'] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({'success': True})


@app.route('/api/train/status', methods=['GET'])
def train_status():
    with training_lock:
        return jsonify(dict(training_state))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
def predict():
    from model_builder import load_saved_model, preprocess_image

    model_dir = app.config['MODEL_FOLDER']
    model_path = os.path.join(model_dir, 'model.keras')
    meta_path = os.path.join(model_dir, 'meta.json')

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return jsonify({'error': 'No trained model found. Please train first.'}), 400

    with open(meta_path) as fh:
        meta = json.load(fh)
    classes = meta['classes']
    img_size = meta.get('img_size', 150)

    # Accept either file upload or base64
    image_data = None
    if 'image' in request.files:
        f = request.files['image']
        image_data = f.read()
    elif request.is_json:
        b64 = request.get_json().get('image', '')
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        image_data = base64.b64decode(b64)
    else:
        return jsonify({'error': 'No image provided'}), 400

    tf = get_tf()
    model = load_saved_model(model_path)
    img_array = preprocess_image(image_data, img_size)
    preds = model.predict(img_array, verbose=0)[0]

    results = [
        {'class': classes[i], 'confidence': float(preds[i])}
        for i in range(len(classes))
    ]
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return jsonify({'predictions': results, 'top': results[0]})


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


# ---------------------------------------------------------------------------
# Bootstrap directories & run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    for d in ['dataset', 'models', 'static', 'templates']:
        os.makedirs(d, exist_ok=True)
    app.run(debug=True, port=5000)
