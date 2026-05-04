"""
model_builder.py
CNN architecture, training loop with live-state updates, and inference helpers.
"""
import os
import io
import json
import numpy as np


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def create_model(num_classes: int, img_size: int = 150):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# Custom Keras callback – writes progress to the shared state dict
# ---------------------------------------------------------------------------
def make_progress_callback(state, lock, total_epochs):
    import tensorflow as tf

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            entry = {
                'epoch': epoch + 1,
                'loss': round(float(logs.get('loss', 0)), 4),
                'accuracy': round(float(logs.get('accuracy', 0)), 4),
                'val_loss': round(float(logs.get('val_loss', 0)), 4),
                'val_accuracy': round(float(logs.get('val_accuracy', 0)), 4),
            }
            with lock:
                state['epoch'] = epoch + 1
                state['loss'] = entry['loss']
                state['accuracy'] = entry['accuracy']
                state['val_loss'] = entry['val_loss']
                state['val_accuracy'] = entry['val_accuracy']
                state['logs'].append(entry)

    return ProgressCallback()


# ---------------------------------------------------------------------------
# Training entry-point called from the background thread in app.py
# ---------------------------------------------------------------------------
def train_model(
    dataset_dir: str,
    model_dir: str,
    static_dir: str,
    epochs: int,
    img_size: int,
    batch_size: int,
    state: dict,
    lock,
):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # ---- Data generators with augmentation --------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )

    classes = list(train_gen.class_indices.keys())
    num_classes = len(classes)

    with lock:
        state['classes'] = classes

    # ---- Build & train model ----------------------------------------------
    model = create_model(num_classes, img_size)

    callbacks = [
        make_progress_callback(state, lock, epochs),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=0
        ),
    ]

    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=0,
    )

    # ---- Save model + metadata --------------------------------------------
    model_path = os.path.join(model_dir, 'model.keras')
    model.save(model_path)

    meta = {'classes': classes, 'img_size': img_size, 'num_classes': num_classes}
    with open(os.path.join(model_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh)

    # ---- Confusion matrix -------------------------------------------------
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    cm_filename = 'confusion_matrix.png'
    cm_path = os.path.join(static_dir, cm_filename)
    fig.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    with lock:
        state['status'] = 'done'
        state['confusion_matrix_path'] = cm_filename


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
_model_cache = {}

def load_saved_model(model_path: str):
    import tensorflow as tf
    if model_path not in _model_cache:
        _model_cache[model_path] = tf.keras.models.load_model(model_path)
    return _model_cache[model_path]


def preprocess_image(image_bytes: bytes, img_size: int = 150):
    """Convert raw image bytes → normalised (1, H, W, 3) numpy array."""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
