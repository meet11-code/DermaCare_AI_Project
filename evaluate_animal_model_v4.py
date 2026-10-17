import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# DERMACARE AI - ANIMAL MODEL V4 EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL SKIN MODEL V4 EVALUATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_clean_split",
    "validation"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "animal",
    "animal_skin_model_v4.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# CHECK PATHS
# ============================================================

print("\nValidation Dataset:")
print(os.path.abspath(VALIDATION_DIR))

print("\nModel:")
print(os.path.abspath(MODEL_PATH))


if not os.path.exists(VALIDATION_DIR):

    print("\nERROR: Validation dataset not found!")

    raise SystemExit


if not os.path.exists(MODEL_PATH):

    print("\nERROR: V4 model not found!")

    raise SystemExit


# ============================================================
# LOAD VALIDATION DATASET
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


class_names = validation_dataset.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 70)
print("LOADING V4 MODEL")
print("-" * 70)


model = tf.keras.models.load_model(
    MODEL_PATH
)


print("\nV4 model loaded successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
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
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# GET TRUE LABELS AND PREDICTIONS
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

    true_labels = np.argmax(
        labels.numpy(),
        axis=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        true_labels
    )

    y_pred.extend(
        predicted_labels
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


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


print("\nRows = Actual Class")
print("Columns = Predicted Class\n")


print(
    " " * 35,
    " ".join(
        f"{i:>5}"
        for i in range(len(class_names))
    )
)


for i, row in enumerate(cm):

    print(
        f"{i:>2} {class_names[i]:<30}",
        " ".join(
            f"{value:>5}"
            for value in row
        )
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    digits=4,

    zero_division=0
)


print("\n")
print(report)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)


for i, class_name in enumerate(class_names):

    actual_count = np.sum(
        y_true == i
    )

    predicted_count = np.sum(
        y_pred == i
    )

    correct_count = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    print(
        f"\n{class_name}"
    )

    print(
        "Actual:",
        actual_count
    )

    print(
        "Predicted:",
        predicted_count
    )

    print(
        "Correct:",
        correct_count
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("V4 EVALUATION COMPLETED")
print("=" * 70)