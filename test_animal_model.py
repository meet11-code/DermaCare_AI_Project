import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# ============================================================
# ANIMAL MODEL TEST
# ============================================================

MODEL_PATH = os.path.join(
    "model",
    "animal",
    "animal_skin_model_v1.keras"
)


CLASS_NAMES = [
    "Bacterial_dermatosis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis"
]


print("\n" + "=" * 70)
print("DERMACARE AI - ANIMAL MODEL TEST")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Animal Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Animal Model Loaded Successfully!")

print(
    "Model Input Shape:",
    model.input_shape
)

print(
    "Model Output Shape:",
    model.output_shape
)


# ============================================================
# ASK IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter animal image path: "
).strip().strip('"')


if not os.path.exists(image_path):

    print("\nERROR: Image file not found!")

    print(
        "Path:",
        image_path
    )

    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading Image...")

img = image.load_img(
    image_path,
    target_size=(224, 224)
)


img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

print("\nPredicting Animal Skin Condition...")

prediction = model.predict(
    img_array,
    verbose=0
)[0]


# ============================================================
# RESULT
# ============================================================

predicted_index = int(
    np.argmax(prediction)
)

confidence = float(
    prediction[predicted_index]
) * 100


print("\n" + "=" * 70)
print("ANIMAL MODEL RESULT")
print("=" * 70)

print(
    "Predicted Class Index:",
    predicted_index
)

print(
    "Predicted Condition:",
    CLASS_NAMES[predicted_index]
)

print(
    "Confidence:",
    f"{confidence:.2f}%"
)


# ============================================================
# ALL CLASS PROBABILITIES
# ============================================================

print("\nClass Probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{i} -> {class_name}: "
        f"{prediction[i] * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)