import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "animal",
    "animal_skin_model_v3.keras"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "validation"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING ANIMAL MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DATASET")
print("=" * 70)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print(
    "\nOverall Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for i, name in enumerate(class_names):

    actual_count = np.sum(
        y_true == i
    )

    predicted_count = np.sum(
        y_pred == i
    )

    print(
        f"\n{name}:"
    )

    print(
        "Actual:",
        actual_count
    )

    print(
        "Predicted:",
        predicted_count
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)