import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# DERMACARE AI - ANIMAL MODEL V2 TEST
# ============================================================

print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL MODEL V2 DETAILED TEST")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "animal",
    "animal_skin_model_v2.keras"
)

VALIDATION_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "animal_combined_split",
    "validation"
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# CHECK PATHS
# ============================================================

print("\nModel:")
print(os.path.abspath(MODEL_PATH))

print("\nValidation Dataset:")
print(os.path.abspath(VALIDATION_DIR))

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found!")
    raise SystemExit

if not os.path.exists(VALIDATION_DIR):
    print("\nERROR: Validation dataset not found!")
    raise SystemExit

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 70)
print("LOADING ANIMAL MODEL V2")
print("-" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Animal Model V2 Loaded Successfully!")

# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "-" * 70)
print("LOADING VALIDATION DATASET")
print("-" * 70)

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

for index, class_name in enumerate(class_names):
    print(index, "->", class_name)

# ============================================================
# PREDICTION
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
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    "Validation Images:",
    len(y_true)
)

print(
    "Correct Predictions:",
    np.sum(y_true == y_pred)
)

print(
    "Wrong Predictions:",
    np.sum(y_true != y_pred)
)

print(
    "Validation Accuracy:",
    round(accuracy * 100, 2),
    "%"
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

print(report)

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
    " " * 32 +
    " ".join(
        f"{i:^10}"
        for i in range(len(class_names))
    )
)

for i, row in enumerate(cm):

    print(
        f"{i:^32}" +
        " ".join(
            f"{value:^10}"
            for value in row
        )
    )

# ============================================================
# CLASS-WISE ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE ACCURACY")
print("=" * 70)

for i, class_name in enumerate(class_names):

    total = np.sum(y_true == i)

    correct = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    class_accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{correct}/{total} "
        f"correct "
        f"({class_accuracy:.2f}%)"
    )

# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("WRONG PREDICTIONS SUMMARY")
print("=" * 70)

wrong_indices = np.where(
    y_true != y_pred
)[0]

print(
    "Total Wrong:",
    len(wrong_indices)
)

print("\nFirst 30 wrong predictions:")

file_paths = validation_dataset.file_paths

for count, index in enumerate(wrong_indices[:30]):

    actual = class_names[
        y_true[index]
    ]

    predicted = class_names[
        y_pred[index]
    ]

    print(
        f"\n{count + 1}. "
        f"{os.path.basename(file_paths[index])}"
    )

    print(
        "   Actual   :",
        actual
    )

    print(
        "   Predicted:",
        predicted
    )

# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ANIMAL MODEL V2 TEST COMPLETED")
print("=" * 70)