import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ==========================================
# LOAD TRAINED VALIDATOR MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "validator_model.keras"
)

print("Loading Validator Model...")

validator_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Validator Model Loaded Successfully!")
print("Model Path:", MODEL_PATH)


# ==========================================
# VALIDATE SKIN IMAGE
# ==========================================

def validate_skin_image(image_path):

    try:

        # ----------------------------------
        # Open Image
        # ----------------------------------

        img = Image.open(image_path).convert("RGB")


        # ----------------------------------
        # Resize Image
        # ----------------------------------

        img = img.resize((224, 224))


        # ----------------------------------
        # Convert Image to NumPy
        # ----------------------------------

        img_array = np.array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # ----------------------------------
        # IMPORTANT
        # ----------------------------------
        # Same preprocessing used during
        # validator model training
        # ----------------------------------

        img_array = img_array / 255.0


        # ----------------------------------
        # Model Prediction
        # ----------------------------------

        prediction = validator_model.predict(
            img_array,
            verbose=0
        )[0]


        # ----------------------------------
        # Get Class
        # ----------------------------------

        predicted_class = np.argmax(
            prediction
        )


        confidence = float(
            prediction[predicted_class]
        ) * 100


        # ----------------------------------
        # Class Mapping
        # ----------------------------------
        # image_dataset_from_directory
        # automatically sorts alphabetically
        #
        # non_skin = 0
        # skin     = 1
        # ----------------------------------

        if predicted_class == 1:

            print(
                "Validator Result: SKIN"
            )

            print(
                "Confidence:",
                round(confidence, 2),
                "%"
            )

            return 1, confidence

        else:

            print(
                "Validator Result: NON-SKIN"
            )

            print(
                "Confidence:",
                round(confidence, 2),
                "%"
            )

            return 0, confidence


    except Exception as e:

        print(
            "Validator Error:",
            e
        )

        return 0, 0.0