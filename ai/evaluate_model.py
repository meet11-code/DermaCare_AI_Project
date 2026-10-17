import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ============================================================
# DERMACARE AI - MODEL EVALUATION
# ============================================================

print("=" * 60)
print("DERMACARE AI - SKIN DISEASE MODEL EVALUATION")
print("=" * 60)

# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "human"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "skin_model.keras"
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Healthy",
    "Melanoma",
    "Psoriasis",
    "Ringworm"
]

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nValidation dataset loaded!")

print(
    "Number of validation batches:",
    len(validation_dataset)
)

# ============================================================
# GET TRUE LABELS AND PREDICTIONS
# ============================================================

print("\nRunning predictions...")
print("Please wait...")

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
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

# ============================================================
# CLASS-WISE ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("CLASS-WISE ACCURACY")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    total_images = np.sum(
        y_true == i
    )

    correct_images = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    if total_images > 0:

        accuracy = (
            correct_images /
            total_images
        ) * 100

    else:

        accuracy = 0

    print(
        f"{class_name}: "
        f"{accuracy:.2f}% "
        f"({correct_images}/{total_images})"
    )

# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = (
    np.sum(y_true == y_pred) /
    len(y_true)
) * 100

print("\n" + "=" * 60)
print("OVERALL VALIDATION ACCURACY")
print("=" * 60)

print(
    f"{overall_accuracy:.2f}%"
)

# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

REPORT_DIR = os.path.join(
    BASE_DIR,
    "static",
    "reports"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

CM_PATH = os.path.join(
    REPORT_DIR,
    "confusion_matrix.png"
)

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "DermaCare AI - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

# Add values inside matrix

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CM_PATH,
    dpi=300
)

plt.close()

print("\nConfusion matrix saved!")

print(
    CM_PATH
)

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION FINISHED")
print("=" * 60)