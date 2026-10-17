import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# DERMACARE AI - ANIMAL SKIN MODEL V2
# TWO-STAGE TRANSFER LEARNING + FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL SKIN MODEL V2 TRAINING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_combined_split",
    "train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_combined_split",
    "validation"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "animal"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "animal_skin_model_v2.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

STAGE1_EPOCHS = 12

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
# LOAD TRAIN DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING TRAINING DATASET")
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


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "-" * 70)
print("LOADING VALIDATION DATASET")
print("-" * 70)

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
    print(index, "->", class_name)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)

validation_dataset = validation_dataset.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.10),

    layers.RandomZoom(0.10),

    layers.RandomContrast(0.10),

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

    input_shape=(224, 224, 3)
)


# ============================================================
# STAGE 1 - FREEZE BASE MODEL
# ============================================================

base_model.trainable = False

print("\nStage 1:")
print("MobileNetV2 Frozen")


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

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

    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.000001,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - CLASSIFIER TRAINING")
print("=" * 70)

history_stage1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE1_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - MOBILENETV2 FINE TUNING")
print("=" * 70)


# Unfreeze the base model
base_model.trainable = True


# Freeze early layers.
# Only the last 30 layers will be trainable.

for layer in base_model.layers[:-30]:
    layer.trainable = False


print(
    "Trainable MobileNetV2 layers:",
    sum(
        layer.trainable
        for layer in base_model.layers
    )
)


# ============================================================
# RECOMPILE WITH VERY LOW LEARNING RATE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# FINE-TUNING TRAINING
# ============================================================

history_stage2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE2_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

loss, accuracy = model.evaluate(

    validation_dataset,

    verbose=1
)


print(
    "\nValidation Loss:",
    round(loss, 4)
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
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL MODEL V2 TRAINING COMPLETED")
print("=" * 70)

print(
    "Model Saved:",
    os.path.abspath(MODEL_PATH)
)

print("\nAnimal Classes:")

for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )


print("\n" + "=" * 70)
print("NEXT STEP: TEST ANIMAL MODEL V2")
print("=" * 70)