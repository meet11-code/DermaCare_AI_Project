import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = r"C:\Users\meett\OneDrive\Desktop\DermaCare_AI\dataset\validator"


# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = r"C:\Users\meett\OneDrive\Desktop\DermaCare_AI\model\validator_model.keras"


# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 15


# ==========================================
# START
# ==========================================

print("=" * 60)

print("DERMACARE AI - VALIDATOR MODEL TRAINING")

print("=" * 60)

print("\nDataset Path:")
print(DATASET_PATH)

print("\nModel Path:")
print(MODEL_PATH)


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Validator Dataset folder not found!")

    print(DATASET_PATH)

    exit()


print("\nValidator Dataset folder found successfully!")


# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    validation_split=0.2,

    subset="training",

    seed=123

)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_PATH,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    validation_split=0.2,

    subset="validation",

    seed=123

)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = train_dataset.class_names


print("\n" + "=" * 60)

print("VALIDATOR CLASSES")

print("=" * 60)


for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )


print("\nTotal Classes:", len(class_names))


# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)


validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# LOAD MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")


base_model = MobileNetV2(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"

)


# Freeze MobileNetV2

base_model.trainable = False


# ==========================================
# CREATE MODEL
# ==========================================

model = models.Sequential([

    # Input image
    layers.Input(
        shape=(224, 224, 3)
    ),

    # MobileNetV2 preprocessing
    layers.Lambda(
        preprocess_input
    ),

    # MobileNetV2
    base_model,

    # Convert feature maps
    layers.GlobalAveragePooling2D(),

    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(
        0.3
    ),

    # Final classification
    layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n" + "=" * 60)

print("VALIDATOR MODEL CREATED")

print("=" * 60)


model.summary()


# ==========================================
# START TRAINING
# ==========================================

print("\n" + "=" * 60)

print("STARTING VALIDATOR TRAINING")

print("=" * 60)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS

)


# ==========================================
# SAVE MODEL
# ==========================================

print("\n" + "=" * 60)

print("VALIDATOR TRAINING COMPLETED")

print("=" * 60)


# Create model folder

os.makedirs(

    os.path.dirname(MODEL_PATH),

    exist_ok=True

)


# Save model

model.save(

    MODEL_PATH

)


print("\nVALIDATOR MODEL SAVED SUCCESSFULLY!")


print("\nLocation:")

print(MODEL_PATH)


# ==========================================
# FINAL CLASS ORDER
# ==========================================

print("\n" + "=" * 60)

print("FINAL VALIDATOR CLASS ORDER")

print("=" * 60)


for index, class_name in enumerate(class_names):

    print(

        index,

        "->",

        class_name

    )


print("\nValidator Training Finished Successfully!")