import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# DERMACARE AI - ANIMAL SKIN MODEL V4
# TWO-STAGE TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL SKIN MODEL V4 TRAINING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "validation"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "animal"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "animal_skin_model_v4.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

STAGE1_EPOCHS = 10

STAGE2_EPOCHS = 15

SEED = 42


# ============================================================
# CHECK DATASET
# ============================================================

print("\nTraining Dataset:")
print(os.path.abspath(TRAIN_DIR))

print("\nValidation Dataset:")
print(os.path.abspath(VALIDATION_DIR))


if not os.path.exists(TRAIN_DIR):

    print("\nERROR: Training dataset not found!")

    raise SystemExit


if not os.path.exists(VALIDATION_DIR):

    print("\nERROR: Validation dataset not found!")

    raise SystemExit


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("LOADING DATASETS")
print("-" * 70)


train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="categorical",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED
)


validation_dataset = tf.keras.utils.image_dataset_from_directory(

    VALIDATION_DIR,

    labels="inferred",

    label_mode="categorical",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)


print("\n" + "=" * 70)
print("ANIMAL CLASS INFORMATION")
print("=" * 70)

print("Total Classes:", NUM_CLASSES)

for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    ),

], name="animal_data_augmentation")


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)


base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    ),

    name="animal_mobilenetv2"
)


# ============================================================
# STAGE 1 - FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


print(
    "\nStage 1 Base Model Trainable:",
    base_model.trainable
)


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


x = preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.BatchNormalization()(
    x
)


x = layers.Dropout(
    0.35
)(
    x
)


x = layers.Dense(
    128,
    activation="relu"
)(
    x
)


x = layers.Dropout(
    0.25
)(
    x
)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax"

)(
    x
)


model = models.Model(

    inputs,

    outputs

)


# ============================================================
# STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)


# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


stage1_early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


stage1_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - FROZEN MOBILENETV2 TRAINING")
print("=" * 70)


history_stage1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE1_EPOCHS,

    callbacks=[

        stage1_checkpoint,

        stage1_early_stopping,

        stage1_reduce_lr

    ]

)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST STAGE 1 MODEL")
print("=" * 70)


model = tf.keras.models.load_model(

    MODEL_PATH

)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - FINE TUNING MOBILENETV2")
print("=" * 70)


# Get the actual MobileNetV2 base model by its name
base_model = model.get_layer("animal_mobilenetv2")


# Freeze all MobileNetV2 layers first
for layer in base_model.layers:
    layer.trainable = False


# Fine-tune only the last 30 MobileNetV2 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True


# Keep BatchNormalization layers frozen
# This makes fine-tuning more stable on a small dataset.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


trainable_count = sum(

    1

    for layer in base_model.layers

    if layer.trainable

)


print(
    "Fine-tunable MobileNetV2 layers:",
    trainable_count
)


# ============================================================
# STAGE 2 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


stage2_early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


stage2_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history_stage2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE2_EPOCHS,

    callbacks=[

        stage2_checkpoint,

        stage2_early_stopping,

        stage2_reduce_lr

    ]

)


# ============================================================
# LOAD BEST FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST V4 MODEL")
print("=" * 70)


best_model = tf.keras.models.load_model(

    MODEL_PATH

)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL V4 MODEL EVALUATION")
print("=" * 70)


loss, accuracy = best_model.evaluate(

    validation_dataset,

    verbose=1

)


print(
    "\nValidation Loss:",
    round(
        loss,
        4
    )
)


print(
    "Validation Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# BEST ACCURACY FROM TRAINING
# ============================================================

best_stage1_accuracy = max(

    history_stage1.history["val_accuracy"]

)


best_stage2_accuracy = max(

    history_stage2.history["val_accuracy"]

)


print(
    "\nBest Stage 1 Validation Accuracy:",
    round(
        best_stage1_accuracy * 100,
        2
    ),
    "%"
)


print(
    "Best Stage 2 Validation Accuracy:",
    round(
        best_stage2_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# FINAL SAVE
# ============================================================

best_model.save(

    MODEL_PATH

)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL MODEL V4 TRAINING COMPLETED")
print("=" * 70)


print(
    "Model Saved:",
    os.path.abspath(
        MODEL_PATH
    )
)


print("\nAnimal Classes:")

for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )


print("\n" + "=" * 70)
print("NEXT STEP: EVALUATE V4")
print("=" * 70)