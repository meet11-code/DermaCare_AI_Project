import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "model/skin_model_v2.keras"

VALIDATION_PATH = "dataset/human_split/validation"

CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Healthy",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]

# ==========================================
# LOAD MODEL
# ==========================================

print("\n" + "=" * 60)
print("DERMACARE AI - MODEL TESTING")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

print("Input shape:", model.input_shape)

print("Output shape:", model.output_shape)


# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATASET")
print("=" * 60)

validation_ds = image_dataset_from_directory(

    VALIDATION_PATH,

    image_size=(224, 224),

    batch_size=32,

    shuffle=False

)

print("\nDataset loaded successfully!")

print(
    "Dataset classes:",
    validation_ds.class_names
)


# ==========================================
# CHECK CLASS ORDER
# ==========================================

if validation_ds.class_names != CLASS_NAMES:

    print("\nWARNING: CLASS ORDER MISMATCH!")

    print(
        "Expected:",
        CLASS_NAMES
    )

    print(
        "Found:",
        validation_ds.class_names
    )

else:

    print("\nClass order is correct!")


# ==========================================
# IMPORTANT
# DO NOT NORMALIZE HERE
# ==========================================

print("\nUsing raw pixel values (0-255).")

print(
    "Model contains internal MobileNetV2 preprocessing."
)


# ==========================================
# PREDICT
# ==========================================

print("\n" + "=" * 60)
print("RUNNING PREDICTIONS")
print("=" * 60)

print("\nPlease wait...\n")

predictions = model.predict(

    validation_ds,

    verbose=1

)


# ==========================================
# PREDICTED CLASSES
# ==========================================

predicted_classes = np.argmax(

    predictions,

    axis=1

)


# ==========================================
# GET TRUE LABELS
# ==========================================

true_classes = np.concatenate(

    [

        y.numpy()

        for x, y in validation_ds

    ]

)


# ==========================================
# CHECK PREDICTION DISTRIBUTION
# ==========================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(

        predicted_classes == i

    )

    print(

        f"{class_name}: {count}"

    )


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(

    classification_report(

        true_classes,

        predicted_classes,

        target_names=CLASS_NAMES,

        zero_division=0

    )

)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(

    true_classes,

    predicted_classes

)

print(cm)


# ==========================================
# FINAL ACCURACY
# ==========================================

accuracy = np.mean(

    true_classes == predicted_classes

)

print("\n" + "=" * 60)
print("FINAL TEST ACCURACY")
print("=" * 60)

print(

    f"Accuracy: {accuracy * 100:.2f}%"

)

print("\n" + "=" * 60)
print("MODEL TESTING COMPLETED")
print("=" * 60)