import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# DERMACARE AI - BALANCED HUMAN SKIN DISEASE MODEL
# ============================================================

print("=" * 60)
print("DERMACARE AI - BALANCED HUMAN SKIN DISEASE MODEL TRAINING")
print("=" * 60)

# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "human"
)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "skin_model.keras"
)

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

SEED = 42

print("\nProject Directory:")
print(BASE_DIR)

print("\nDataset Path:")
print(DATASET_PATH)

print("\nModel Path:")
print(MODEL_PATH)

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset folder not found!")

    print(DATASET_PATH)

    exit()

print("\nDataset folder found successfully!")

# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=VALIDATION_SPLIT,

    subset="training",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True

)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATASET")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    validation_split=VALIDATION_SPLIT,

    subset="validation",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n" + "=" * 60)
print("CLASSES FOUND")
print("=" * 60)

for i, name in enumerate(class_names):

    print(
        f"{i} -> {name}"
    )

print(
    "\nTotal Classes:",
    len(class_names)
)

# ============================================================
# EXPECTED CLASS ORDER
# ============================================================

expected_classes = [

    "Acne",

    "Eczema",

    "Healthy",

    "Melanoma",

    "Psoriasis",

    "Ringworm"

]

if class_names != expected_classes:

    print("\nWARNING!")

    print(
        "Class order is different!"
    )

    print(
        "Expected:",
        expected_classes
    )

    print(
        "Found:",
        class_names
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
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("CREATING CLASS WEIGHTS")
print("=" * 60)

# Dataset image counts

class_counts = {

    0: 1833,  # Acne

    1: 1677,  # Eczema

    2: 4496,  # Healthy

    3: 3140,  # Melanoma

    4: 2055,  # Psoriasis

    5: 1702   # Ringworm

}

total_images = sum(
    class_counts.values()
)

num_classes = len(
    class_counts
)

class_weight = {}

for class_id, count in class_counts.items():

    weight = (
        total_images
        /
        (
            num_classes
            *
            count
        )
    )

    class_weight[class_id] = weight

    print(
        f"Class {class_id} "
        f"({class_names[class_id]}) "
        f"Weight: {weight:.3f}"
    )

# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING DATA AUGMENTATION")
print("=" * 60)

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"

)

# ============================================================
# MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

base_model = MobileNetV2(

    input_shape=(

        IMG_SIZE[0],

        IMG_SIZE[1],

        3

    ),

    include_top=False,

    weights="imagenet"

)

# Freeze MobileNetV2

base_model.trainable = False

print(
    "\nMobileNetV2 Loaded Successfully!"
)

# ============================================================
# CREATE MODEL
# ============================================================

model = models.Sequential(

    [

        layers.Input(

            shape=(

                IMG_SIZE[0],

                IMG_SIZE[1],

                3

            )

        ),

        # Data Augmentation

        data_augmentation,

        # MobileNetV2 preprocessing

        layers.Rescaling(

            1.0 / 127.5,

            offset=-1

        ),

        # MobileNetV2

        base_model,

        # Feature Extraction

        layers.GlobalAveragePooling2D(),

        # Dense Layer

        layers.Dense(

            256,

            activation="relu"

        ),

        # Dropout

        layers.Dropout(

            0.4

        ),

        # Output

        layers.Dense(

            len(class_names),

            activation="softmax"

        )

    ]

)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL CREATED SUCCESSFULLY"
)

print("=" * 60)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=4,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=0.000001,

    verbose=1

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)

print(
    "STARTING BALANCED TRAINING"
)

print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=15,

    class_weight=class_weight,

    callbacks=[

        early_stopping,

        reduce_lr,

        checkpoint

    ]

)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)

print(
    "LOADING BEST MODEL"
)

print("=" * 60)

best_model = tf.keras.models.load_model(

    MODEL_PATH

)

# ============================================================
# FINAL SAVE
# ============================================================

best_model.save(

    MODEL_PATH

)

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)

print(
    "MODEL TRAINING COMPLETED"
)

print("=" * 60)

print("\nModel Saved:")

print(
    MODEL_PATH
)

print("\nFinal Class Order:")

for i, name in enumerate(
    class_names
):

    print(
        f"{i} -> {name}"
    )

print(

    "\nTraining Finished Successfully!"

)