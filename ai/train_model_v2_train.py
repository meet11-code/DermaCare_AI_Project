import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# DERMACARE AI - IMPROVED HUMAN SKIN DISEASE MODEL TRAINING
# ============================================================

print("=" * 60)
print("DERMACARE AI - IMPROVED MODEL TRAINING")
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
# DATASET PATHS
# ============================================================

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "human_split",
    "train"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "human_split",
    "validation"
)

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "skin_model_v2.keras"
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
SEED = 42

# ============================================================
# PRINT PATHS
# ============================================================

print("\nProject Directory:")
print(BASE_DIR)

print("\nTraining Dataset:")
print(TRAIN_DIR)

print("\nValidation Dataset:")
print(VALIDATION_DIR)

print("\nModel Path:")
print(MODEL_PATH)

# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):

    print("\nERROR: Training dataset not found!")
    exit()

if not os.path.exists(VALIDATION_DIR):

    print("\nERROR: Validation dataset not found!")
    exit()

print(
    "\nTraining and validation datasets found successfully!"
)

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATASET")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED
)

# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATASET")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    VALIDATION_DIR,

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

for index, class_name in enumerate(class_names):

    print(
        f"{index} -> {class_name}"
    )

print(
    "\nTotal Classes:",
    len(class_names)
)

# ============================================================
# EXPECTED CLASSES
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
        "Class order does not match expected order."
    )

    print(
        "\nExpected:",
        expected_classes
    )

    print(
        "\nFound:",
        class_names
    )

    print(
        "\nPlease check dataset folders."
    )

    exit()

# ============================================================
# CLASS COUNTS
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING CLASS WEIGHTS")
print("=" * 60)

class_counts = {}

for index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    count = len([
        f
        for f in os.listdir(class_dir)
        if os.path.isfile(
            os.path.join(
                class_dir,
                f
            )
        )
    ])

    class_counts[index] = count

    print(
        f"{index} -> {class_name}: {count} images"
    )

total_images = sum(
    class_counts.values()
)

num_classes = len(
    class_names
)

# Calculate balanced class weights

class_weights = {}

for class_index, count in class_counts.items():

    class_weights[class_index] = (
        total_images /
        (
            num_classes *
            count
        )
    )

print("\nClass Weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_index} -> "
        f"{class_names[class_index]}: "
        f"{weight:.4f}"
    )

# ============================================================
# PERFORMANCE OPTIMIZATION
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

print("\n" + "=" * 60)
print("CREATING DATA AUGMENTATION")
print("=" * 60)

data_augmentation = tf.keras.Sequential(

    [

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
        )

    ],

    name="data_augmentation"
)

print(
    "Data augmentation created successfully!"
)

# ============================================================
# LOAD MOBILENETV2
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

# Freeze base model

base_model.trainable = False

print(
    "\nMobileNetV2 loaded successfully!"
)

# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = models.Sequential(

    [

        layers.Input(

            shape=(

                IMG_SIZE[0],

                IMG_SIZE[1],

                3

            )

        ),

        # Data augmentation

        data_augmentation,

        # MobileNetV2 preprocessing
        # Converts [0,255] -> [-1,1]

        layers.Rescaling(

            1.0 / 127.5,

            offset=-1

        ),

        # MobileNetV2

        base_model,

        # Feature extraction

        layers.GlobalAveragePooling2D(),

        # Dense layer

        layers.Dense(

            256,

            activation="relu"

        ),

        # Dropout

        layers.Dropout(

            0.4

        ),

        # Output layer

        layers.Dense(

            len(class_names),

            activation="softmax"

        )

    ]

)

print(
    "\nModel created successfully!"
)

# ============================================================
# COMPILE MODEL
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
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

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
# PHASE 1
# TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 60)

print(
    "\nTraining with class weights..."
)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10,

    class_weight=class_weights,

    callbacks=[

        early_stopping,

        reduce_lr,

        checkpoint

    ]

)

# ============================================================
# PHASE 2
# FINE TUNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)

# Unfreeze base model

base_model.trainable = True

# Freeze all except last 30 layers

for layer in base_model.layers[:-30]:

    layer.trainable = False

print(
    "\nFine-tuning last 30 MobileNetV2 layers..."
)

# ============================================================
# RECOMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.00001

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)

# ============================================================
# FINE TUNING
# ============================================================

fine_tune_history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=10,

    class_weight=class_weights,

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
print("LOADING BEST MODEL")
print("=" * 60)

best_model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False

)

# ============================================================
# FINAL SAVE
# ============================================================

best_model.save(

    MODEL_PATH

)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModel saved at:")

print(MODEL_PATH)

# ============================================================
# FINAL CLASS ORDER
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLASS ORDER")
print("=" * 60)

for index, class_name in enumerate(
    class_names
):

    print(
        f"{index} -> {class_name}"
    )

print(
    "\nTotal Classes:",
    len(class_names)
)

print("\n" + "=" * 60)
print("DERMACARE AI MODEL V2 READY!")
print("=" * 60)