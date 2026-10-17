import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# DERMA CARE AI - ANIMAL SKIN MODEL V3
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "validation"
)

MODEL_DIR = os.path.join(BASE_DIR, "model", "animal")
MODEL_PATH = os.path.join(MODEL_DIR, "animal_skin_model_v3.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("DERMACARE AI - ANIMAL SKIN MODEL V3 TRAINING")
print("=" * 70)

print("\nTraining Dataset:")
print(TRAIN_DIR)

print("\nValidation Dataset:")
print(VAL_DIR)

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("LOADING TRAINING DATASET")
print("-" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\n" + "-" * 70)
print("LOADING VALIDATION DATASET")
print("-" * 70)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\n" + "=" * 70)
print("ANIMAL CLASS INFORMATION")
print("=" * 70)

print("Total Classes:", num_classes)

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_counts = []

for class_name in class_names:
    class_path = os.path.join(TRAIN_DIR, class_name)

    count = len([
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f))
    ])

    class_counts.append(count)

    print(f"{class_name}: {count}")

class_indices = []

for i, count in enumerate(class_counts):
    class_indices.extend([i] * count)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.array(class_indices)
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass Weights:")

for i, weight in class_weights.items():
    print(f"{class_names[i]}: {weight:.4f}")

# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA AUGMENTATION")
print("=" * 70)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.10),
    ],
    name="animal_data_augmentation_v3"
)

# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# ============================================================
# STAGE 1
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - CLASSIFIER TRAINING")
print("=" * 70)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks_stage1 = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# TRAIN STAGE 1
# ============================================================

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks_stage1
)

# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - MOBILENETV2 FINE TUNING")
print("=" * 70)

base_model.trainable = True

# Freeze most layers.
# Only last 40 layers will train.

for layer in base_model.layers[:-40]:
    layer.trainable = False

for layer in base_model.layers[-40:]:
    layer.trainable = True

# Keep BatchNormalization frozen for stable fine tuning.

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print("Trainable MobileNetV2 layers:", trainable_count)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=5e-6
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage2 = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks_stage2
)

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

best_model = keras.models.load_model(MODEL_PATH)

loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1
)

print("\nValidation Loss:", round(loss, 4))
print("Validation Accuracy:", round(accuracy * 100, 2), "%")

# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

class_file = os.path.join(
    MODEL_DIR,
    "animal_classes_v3.txt"
)

with open(class_file, "w", encoding="utf-8") as f:
    for i, class_name in enumerate(class_names):
        f.write(f"{i} -> {class_name}\n")

print("\n" + "=" * 70)
print("ANIMAL MODEL V3 TRAINING COMPLETED")
print("=" * 70)

print("\nModel Saved:")
print(MODEL_PATH)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

print("\nClass information saved:")
print(class_file)

print("\n" + "=" * 70)