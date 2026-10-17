import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
from utils.combined_analysis import combine_analysis
import json
from dotenv import load_dotenv
from google import genai
from flask import Flask, render_template, request, redirect, flash
import os
import time
import uuid
import sqlite3

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    url_for,
    send_from_directory,
    flash
)

#import os
import time
import uuid
import sqlite3

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

from PIL import Image, UnidentifiedImageError

from tensorflow.keras.models import load_model


from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(
    ENV_PATH,
    override=True
)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

print("=" * 60)
print("GEMINI KEY TEST")
print("ENV PATH:", ENV_PATH)
print("ENV EXISTS:", os.path.isfile(ENV_PATH))
print("KEY LOADED:", bool(GOOGLE_API_KEY))
print(
    "KEY LENGTH:",
    len(GOOGLE_API_KEY)
    if GOOGLE_API_KEY
    else 0
)
print("=" * 60)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. Check your .env file."
    )

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)

# IMPORTANT:
# Change this secret key in production
app.secret_key = "dermacare_ai_secret_key_change_this"

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

PDF_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "reports"
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "model"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "users.db"
)


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PDF_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# ALLOWED IMAGE EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {

    "png",

    "jpg",

    "jpeg",

    "webp"

}


# ============================================================
# MODEL PATHS
# ============================================================

SKIN_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "skin_model_v2.keras"
)

VALIDATOR_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "validator_model.keras"
)

# ============================================================
# LOAD VALIDATOR MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING SKIN VALIDATOR MODEL")
print("=" * 70)

validator_model = None

if os.path.exists(
    VALIDATOR_MODEL_PATH
):

    try:

        print(
            "Validator Model Found:",
            os.path.abspath(
                VALIDATOR_MODEL_PATH
            )
        )

        validator_model = tf.keras.models.load_model(
            VALIDATOR_MODEL_PATH,
            custom_objects={
                "preprocess_input": preprocess_input
            },
            compile=False
        )

        print(
            "Validator Model Loaded Successfully!"
        )

        print(
            "Validator Model Input Shape:",
            validator_model.input_shape
        )

        print(
            "Validator Model Output Shape:",
            validator_model.output_shape
        )

    except Exception as e:

        print(
            "ERROR: Could not load Validator Model."
        )

        print(
            "Validator Model Error:",
            e
        )

else:

    print(
        "WARNING: Validator Model Not Found!"
    )

    print(
        "Expected Path:",
        os.path.abspath(
            VALIDATOR_MODEL_PATH
        )
    )


# ============================================================
# AI MODEL SETTINGS
# ============================================================

# Disease confidence threshold
DISEASE_CONFIDENCE_THRESHOLD = 60.0

# Validator confidence threshold
# Change this only if your validator model requires it.
VALIDATOR_CONFIDENCE_THRESHOLD = 60.0

# Expected image size
IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# APPLICATION STARTING INFORMATION
# ============================================================

print("=" * 70)
print("DERMACARE AI STARTING")
print("=" * 70)

print(
    "Project Directory:",
    BASE_DIR
)

print(
    "Database:",
    DATABASE_PATH
)

print(
    "Upload Directory:",
    UPLOAD_FOLDER
)

print(
    "Reports Directory:",
    PDF_FOLDER
)

print(
    "Skin Model:",
    SKIN_MODEL_PATH
)

print(
    "Validator Model:",
    VALIDATOR_MODEL_PATH
)

print("=" * 70)


# ============================================================
# LOAD SKIN DISEASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING SKIN DISEASE MODEL")
print("=" * 70)

model = None

if os.path.exists(
    SKIN_MODEL_PATH
):

    try:

        print(
            "Skin Model Found:",
            os.path.abspath(
                SKIN_MODEL_PATH
            )
        )

        model = tf.keras.models.load_model(
            SKIN_MODEL_PATH,
            custom_objects={
              "preprocess_input": preprocess_input
            },
            compile=False
        )

        print(
            "Skin Disease Model Loaded Successfully!"
        )

    except Exception as e:

        print(
            "ERROR: Could not load Skin Disease Model."
        )

        print(
            "Error:",
            e
        )

else:

    print(
        "WARNING: Skin Disease Model Not Found!"
    )

    print(
        "Expected Path:",
        os.path.abspath(
            SKIN_MODEL_PATH
        )
    )


# ============================================================
# VALIDATE SKIN IMAGE
# ============================================================

def validate_skin_image(image_path):

    print("\n" + "=" * 70)
    print("SKIN IMAGE VALIDATION")
    print("=" * 70)

    print(
        "Image Path:",
        os.path.abspath(image_path)
    )

    # ========================================================
    # CHECK FILE EXISTS
    # ========================================================

    if not os.path.exists(image_path):

        print(
            "ERROR: Image file does not exist!"
        )

        return 0, 0.0


    # ========================================================
    # CHECK VALIDATOR MODEL
    # ========================================================

    if validator_model is None:

        print(
            "ERROR: Validator model is not loaded!"
        )

        return 0, 0.0


    try:

        # ====================================================
        # CHECK IMAGE
        # ====================================================

        img = Image.open(image_path)

        print(
            "Image File Validation: SUCCESS"
        )

        print(
            "Image Format:",
            img.format
        )

        print(
            "Image Size:",
            img.size
        )

        print(
            "Image Mode:",
            img.mode
        )


        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        img = img.convert("RGB")

        img = img.resize(
            IMAGE_SIZE
        )

        img_array = np.array(
            img,
            dtype=np.float32
        )

        # Add batch dimension

        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        print(
            "Validator Input Shape:",
            img_array.shape
        )


        # ====================================================
        # VALIDATOR PREDICTION
        # ====================================================

        prediction = validator_model.predict(

            img_array,

            verbose=0

        )


        print(
            "Raw Validator Prediction:",
            prediction
        )


        # ====================================================
        # GET PREDICTED CLASS
        # ====================================================

        predicted_class = int(
            np.argmax(
                prediction[0]
            )
        )


        # ====================================================
        # GET CONFIDENCE
        # ====================================================

        confidence = float(
            np.max(
                prediction[0]
            )
        ) * 100


        # ====================================================
        # VALIDATOR CLASS ORDER
        #
        # 0 = non_skin
        # 1 = skin
        # ====================================================

        if predicted_class == 1:

            # ================================================
            # SKIN IMAGE
            # ================================================

            print(
                "Validator Prediction: SKIN"
            )

            print(
                "Predicted Class Index:",
                predicted_class
            )

            print(
                "Validator Confidence:",
                round(
                    confidence,
                    2
                ),
                "%"
            )

            return 1, confidence


        else:

            # ================================================
            # NON-SKIN IMAGE
            # ================================================

            print(
                "Validator Prediction: NON-SKIN"
            )

            print(
                "Predicted Class Index:",
                predicted_class
            )

            print(
                "Validator Confidence:",
                round(
                    confidence,
                    2
                ),
                "%"
            )

            return 0, confidence


    except Exception as e:

        print(
            "Validator Prediction Error:",
            e
        )

        return 0, 0.0


# ============================================================
# HUMAN SKIN DISEASE CLASS ORDER
#
# IMPORTANT:
# THIS ORDER MUST EXACTLY MATCH TRAINING CLASS ORDER
# ============================================================

human_classes = [

    "Acne",

    "Eczema",

    "Healthy",

    "Melanoma",

    "Psoriasis",

    "Ringworm"

]


print("\n" + "=" * 70)
print("HUMAN DISEASE CLASSES")
print("=" * 70)

for index, class_name in enumerate(
    human_classes
):

    print(
        f"{index} -> {class_name}"
    )

print(
    "Total Classes:",
    len(
        human_classes
    )
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    "Acne": {

        "description":
        "Acne is a common skin condition that can occur when hair follicles become clogged with oil and dead skin cells.",

        "medicine":
        "Treatment depends on the type and severity of acne. Products containing ingredients such as benzoyl peroxide or salicylic acid may be used in some cases. Consult a healthcare professional for appropriate treatment.",

        "remedy":
        "Gently clean the affected area, avoid squeezing pimples, and use non-comedogenic skincare products.",

        "doctor":
        "Consult a dermatologist if acne is severe, painful, persistent, or causing scarring."

    },


    "Eczema": {

        "description":
        "Eczema is a group of conditions that can cause dry, itchy, irritated, or inflamed skin.",

        "medicine":
        "Treatment may include moisturizers and doctor-recommended medicines depending on severity. Consult a healthcare professional before using medication.",

        "remedy":
        "Use a gentle moisturizer regularly and avoid harsh soaps, fragrances, or known skin irritants.",

        "doctor":
        "Consult a dermatologist if symptoms become severe, infected, or persistent."

    },


    "Healthy": {

        "description":
        "The uploaded image appears to show healthy skin with no major visible skin condition detected by the AI model.",

        "medicine":
        "No medicine recommendation is provided based only on this AI prediction.",

        "remedy":
        "Maintain regular skin hygiene, use suitable moisturizer, and protect your skin from excessive sunlight.",

        "doctor":
        "Consult a dermatologist if you notice unusual, changing, painful, or persistent skin changes."

    },


    "Melanoma": {

        "description":
        "The AI model detected a visual pattern that may be associated with melanoma. This result is not a medical diagnosis.",

        "medicine":
        "Do not self-medicate based on this AI result. Professional medical evaluation is strongly recommended.",

        "remedy":
        "Avoid excessive sun exposure and do not delay professional medical evaluation.",

        "doctor":
        "Please consult a dermatologist or qualified healthcare professional as soon as possible for proper examination."

    },


    "Psoriasis": {

        "description":
        "Psoriasis is a chronic skin condition that can cause thick, red, scaly, or inflamed patches on the skin.",

        "medicine":
        "Treatment depends on the severity and type of psoriasis and should be recommended by a qualified healthcare professional.",

        "remedy":
        "Keep the skin moisturized and try to avoid known triggers or irritants.",

        "doctor":
        "Consult a dermatologist for proper diagnosis and treatment."

    },


    "Ringworm": {

        "description":
        "Ringworm is a fungal skin infection that can cause an itchy, circular, or ring-shaped rash.",

        "medicine":
        "Antifungal treatment may be required. Consult a healthcare professional for appropriate diagnosis and treatment.",

        "remedy":
        "Keep the affected area clean and dry and avoid sharing towels, clothing, or personal items.",

        "doctor":
        "Consult a dermatologist if the infection spreads, becomes severe, or does not improve."

    },


    "Uncertain": {

        "description":
        "The AI model could not confidently identify the skin condition from the uploaded image. This result should not be considered a medical diagnosis.",

        "medicine":
        "Do not start any medicine based only on this AI result. Consult a qualified healthcare professional for proper evaluation.",

        "remedy":
        "Keep the affected area clean and avoid applying unknown creams or unverified treatments until the condition is properly evaluated.",

        "doctor":
        "Please consult a dermatologist or qualified healthcare professional for proper examination and diagnosis."

    },


    "Invalid Image": {

        "description":
        "The uploaded image could not be validated as a supported human skin image.",

        "medicine":
        "No medicine recommendation is provided because the image was not validated as a supported skin image.",

        "remedy":
        "Please upload a clear, focused image of the human skin area that you want the AI system to analyze.",

        "doctor":
        "If you have a skin-related concern, consult a qualified dermatologist or healthcare professional."

    }

}


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    # Allows accessing columns by name
    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CHECK ALLOWED FILE
# ============================================================

def allowed_file(
    filename
):

    return (

        "." in filename

        and

        filename.rsplit(
            ".",
            1
        )[1].lower()

        in

        ALLOWED_EXTENSIONS

    )


# ============================================================
# GENERATE UNIQUE IMAGE FILENAME
# ============================================================

def generate_unique_filename(
    original_filename
):

    safe_name = secure_filename(
        original_filename
    )

    extension = ""

    if "." in safe_name:

        extension = (

            "."

            +

            safe_name.rsplit(
                ".",
                1
            )[1].lower()

        )

    unique_name = (

        f"{uuid.uuid4().hex}"

        f"{extension}"

    )

    return unique_name


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path
):

    print(
        "\nPreprocessing Image..."
    )

    try:

        # Open image
        img = Image.open(
            image_path
        )

        # Convert to RGB
        img = img.convert(
            "RGB"
        )

        # Resize image
        img = img.resize(
            IMAGE_SIZE
        )

        # Convert to NumPy
        img_array = np.array(
            img,
            dtype=np.float32
        )

        # ====================================================
        # IMPORTANT
        # ====================================================
        # Current preprocessing:
        # Pixel values remain 0-255.
        #
        # If your model was trained using:
        #
        # image / 255.0
        #
        # then uncomment:
        #
        # img_array = img_array / 255.0
        #
        # If trained using:
        #
        # (image / 127.5) - 1
        #
        # then use:
        #
        # img_array = (img_array / 127.5) - 1.0
        #
        # MUST MATCH TRAINING PREPROCESSING.
        # ====================================================

        # img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        print(
            "Preprocessed Image Shape:",
            img_array.shape
        )

        return img_array

    except Exception as e:

        print(
            "Image Preprocessing Error:",
            e
        )

        raise


# ============================================================
# PREDICT HUMAN SKIN DISEASE
# ============================================================

def predict_human_disease(
    image_path
):

    print("\n" + "=" * 70)
    print("PREDICTING HUMAN SKIN DISEASE")
    print("=" * 70)


    # ========================================================
    # MODEL CHECK
    # ========================================================

    if model is None:

        print(
            "ERROR: Skin disease model is not loaded!"
        )

        return (
            "Prediction Error",
            0.0
        )


    try:
        
        # ====================================================
        # VALIDATE SKIN IMAGE FIRST
        # ====================================================

        is_skin, validator_confidence = validate_skin_image(image_path)

        if is_skin == 0:

            print("Image is NOT a skin image.")

            return (
                "Non-Skin Image",
                validator_confidence
            )

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        img_array = preprocess_image(
            image_path
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(

            img_array,

            verbose=0

        )


        print(
            "Disease Prediction Array:",
            prediction
        )


        # ====================================================
        # PREDICTED CLASS INDEX
        # ====================================================

        predicted_index = int(

            np.argmax(
                prediction
            )

        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = float(

            np.max(
                prediction
            )

            *

            100

        )


        print(
            "Predicted Class Index:",
            predicted_index
        )

        print(
            "Disease Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )


        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if confidence < DISEASE_CONFIDENCE_THRESHOLD:

            print(
                "WARNING: Disease confidence is below threshold."
            )

            return (
                "Uncertain",
                round(
                    confidence,
                    2
                )
            )


        # ====================================================
        # CLASS INDEX SAFETY CHECK
        # ====================================================

        if (

            predicted_index < 0

            or

            predicted_index >= len(
                human_classes
            )

        ):

            print(
                "ERROR: Predicted class index is invalid."
            )

            return (
                "Prediction Error",
                0.0
            )


        # ====================================================
        # GET DISEASE NAME
        # ====================================================

        disease = human_classes[

            predicted_index

        ]


        print(
            "Predicted Disease:",
            disease
        )


        return (

            disease,

            round(
                confidence,
                2
            )

        )


    except Exception as e:

        print(
            "Disease Prediction Error:",
            e
        )

        return (
            "Prediction Error",
            0.0
        )
        
# ============================================================
# TEXT ANALYSIS - BASIC VERSION
# ============================================================

def analyze_skin_problem_text(problem_text):

    problem_text = problem_text.lower().strip()

    if not problem_text:
        return None
    
    return analyze_skin_text(problem_text)

    # ========================================================
    # ACNE KEYWORDS
    # ========================================================

    acne_keywords = [
        "acne",
        "pimple",
        "pimples",
        "pimple",
        "blackhead",
        "whitehead",
        "zits"
    ]

    # ========================================================
    # ECZEMA KEYWORDS
    # ========================================================

    eczema_keywords = [
        "eczema",
        "dry skin",
        "itchy skin",
        "itching",
        "red itchy",
        "skin irritation",
        "irritated skin"
    ]

    # ========================================================
    # PSORIASIS KEYWORDS
    # ========================================================

    psoriasis_keywords = [
        "psoriasis",
        "scaly skin",
        "scaly patches",
        "thick red patches",
        "silver scales"
    ]

    # ========================================================
    # RINGWORM KEYWORDS
    # ========================================================

    ringworm_keywords = [
        "ringworm",
        "ring shaped",
        "ring-shaped",
        "circular rash",
        "round rash",
        "fungal infection"
    ]

    # ========================================================
    # MELANOMA KEYWORDS
    # ========================================================

    melanoma_keywords = [
        "melanoma",
        "changing mole",
        "mole is changing",
        "dark mole",
        "new mole",
        "bleeding mole",
        "irregular mole"
    ]

    # ========================================================
    # CHECK ACNE
    # ========================================================

    if any(
        keyword in problem_text
        for keyword in acne_keywords
    ):

        return "Acne"

    # ========================================================
    # CHECK ECZEMA
    # ========================================================

    if any(
        keyword in problem_text
        for keyword in eczema_keywords
    ):

        return "Eczema"

    # ========================================================
    # CHECK PSORIASIS
    # ========================================================

    if any(
        keyword in problem_text
        for keyword in psoriasis_keywords
    ):

        return "Psoriasis"

    # ========================================================
    # CHECK RINGWORM
    # ========================================================

    if any(
        keyword in problem_text
        for keyword in ringworm_keywords
    ):

        return "Ringworm"

    # ========================================================
    # CHECK MELANOMA
    # ========================================================

    if any(
        keyword in problem_text
        for keyword in melanoma_keywords
    ):

        return "Melanoma"

    # ========================================================
    # NO MATCH
    # ========================================================

    return "Uncertain"

# ============================================================
# GET DISEASE INFORMATION
# ============================================================

def get_disease_information(
    disease
):

    if disease in disease_info:

        return disease_info[
            disease
        ]


    return {

        "description":
        "No information is available for this AI prediction.",

        "medicine":
        "Please consult a qualified healthcare professional.",

        "remedy":
        "Avoid self-treatment based only on an AI prediction.",

        "doctor":
        "Consult a dermatologist or qualified healthcare professional."

    }


# ============================================================
# CREATE PDF REPORT
# ============================================================

def create_pdf(
username,
disease,
confidence,
description,
medicine,
remedy,
doctor,
problem_text=None,
combined_result=None
):

    # ========================================================
    # SAFE USERNAME
    # ========================================================

    safe_username = secure_filename(
        str(username)
    )

    if not safe_username:
        safe_username = "user"


    # ========================================================
    # UNIQUE PDF NAME
    # ========================================================

    timestamp = int(time.time())

    unique_id = uuid.uuid4().hex[:8]

    pdf_filename = (
        f"medical_report_"
        f"{safe_username}_"
        f"{timestamp}_"
        f"{unique_id}.pdf"
    )

    pdf_path = os.path.join(
        PDF_FOLDER,
        pdf_filename
    )


    # ========================================================
    # CREATE PDF
    # ========================================================

    c = canvas.Canvas(
        pdf_path,
        pagesize=letter
    )

    page_width, page_height = letter

    left_margin = 50
    right_margin = 50

    text_width = (
        page_width
        - left_margin
        - right_margin
    )

    y = 760


    # ========================================================
    # TEXT WRAPPING
    # ========================================================

    def draw_wrapped_text(
        text,
        x,
        current_y,
        width,
        font_name="Helvetica",
        font_size=11,
        line_height=16
    ):

        c.setFont(
            font_name,
            font_size
        )

        words = str(
            text or "-"
        ).split()

        lines = []

        current_line = ""

        for word in words:

            test_line = (
                current_line
                + " "
                + word
            ).strip()

            test_width = stringWidth(
                test_line,
                font_name,
                font_size
            )

            if test_width <= width:

                current_line = test_line

            else:

                if current_line:

                    lines.append(
                        current_line
                    )

                current_line = word


        if current_line:

            lines.append(
                current_line
            )


        for line in lines:

            if current_y < 70:

                c.showPage()

                current_y = 760

                c.setFont(
                    font_name,
                    font_size
                )


            c.drawString(
                x,
                current_y,
                line
            )

            current_y -= line_height


        return current_y


    # ========================================================
    # SECTION HEADING
    # ========================================================

    def draw_heading(
        title,
        current_y
    ):

        if current_y < 100:

            c.showPage()

            current_y = 760


        c.setFont(
            "Helvetica-Bold",
            12
        )

        c.drawString(
            left_margin,
            current_y,
            title
        )

        return current_y - 20


    # ========================================================
    # TITLE
    # ========================================================

    c.setFont(
        "Helvetica-Bold",
        20
    )

    c.drawCentredString(
        page_width / 2,
        y,
        "DermaCare AI"
    )

    y -= 25


    c.setFont(
        "Helvetica",
        11
    )

    c.drawCentredString(
        page_width / 2,
        y,
        "AI-Assisted Skin Analysis Report"
    )

    y -= 45


    # ========================================================
    # USER INFORMATION
    # ========================================================

    c.setFont(
        "Helvetica",
        12
    )

    c.drawString(
        left_margin,
        y,
        f"User : {username}"
    )

    y -= 25


    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    y = draw_heading(
        "Image Analysis",
        y
    )


    c.setFont(
        "Helvetica",
        11
    )

    c.drawString(
        left_margin + 20,
        y,
        f"Prediction : {disease}"
    )

    y -= 20


    c.drawString(
        left_margin + 20,
        y,
        f"Confidence : {confidence}%"
    )

    y -= 30


    # ========================================================
    # USER SYMPTOMS
    # ========================================================

    if problem_text:

        y = draw_heading(
            "Reported Symptoms / Problem",
            y
        )

        y = draw_wrapped_text(
            problem_text,
            left_margin + 20,
            y,
            text_width - 20
        )

        y -= 20


    # ========================================================
    # COMBINED AI ANALYSIS
    # ========================================================

    if combined_result:

        y = draw_heading(
            "Combined AI Analysis",
            y
        )


        # ----------------------------------------------------
        # FINAL DISEASE
        # ----------------------------------------------------

        final_disease = combined_result.get(
            "final_disease",
            "Inconclusive"
        )

        y = draw_wrapped_text(
            f"Final Assessment : {final_disease}",
            left_margin + 20,
            y,
            text_width - 20
        )

        y -= 5


        # ----------------------------------------------------
        # MATCH LEVEL
        # ----------------------------------------------------

        match_level = combined_result.get(
            "match_level",
            "INCONCLUSIVE"
        )

        y = draw_wrapped_text(
            f"Image & Symptoms Match : {match_level}",
            left_margin + 20,
            y,
            text_width - 20
        )

        y -= 5


        # ----------------------------------------------------
        # COMBINED CONFIDENCE
        # ----------------------------------------------------

        combined_confidence = combined_result.get(
            "confidence",
            "Low"
        )

        y = draw_wrapped_text(
            f"Combined Confidence : {combined_confidence}",
            left_margin + 20,
            y,
            text_width - 20
        )

        y -= 15
        
        try:

            confidence = float(
                str(combined_confidence)
                .replace("%", "")
                .strip()
            )

        except Exception:

            confidence = 0.0


        # ----------------------------------------------------
        # AI ASSESSMENT
        # ----------------------------------------------------

        assessment = combined_result.get(
            "assessment",
            ""
        )

        if assessment:

            y = draw_heading(
                "AI Assessment",
                y
            )

            y = draw_wrapped_text(
                assessment,
                left_margin + 20,
                y,
                text_width - 20
            )

            y -= 15


        # ----------------------------------------------------
        # RECOMMENDED CARE
        # ----------------------------------------------------

        combined_care = combined_result.get(
            "care",
            ""
        )

        if combined_care:

            y = draw_heading(
                "Recommended Care",
                y
            )

            y = draw_wrapped_text(
                combined_care,
                left_margin + 20,
                y,
                text_width - 20
            )

            y -= 15


        # ----------------------------------------------------
        # COMBINED MEDICINE
        # ----------------------------------------------------

        combined_medicine = combined_result.get(
            "medicine",
            ""
        )

        if combined_medicine:

            y = draw_heading(
                "Combined Medicine Information",
                y
            )

            y = draw_wrapped_text(
                combined_medicine,
                left_margin + 20,
                y,
                text_width - 20
            )

            y -= 15


        # ----------------------------------------------------
        # COMBINED DOCTOR ADVICE
        # ----------------------------------------------------

        combined_doctor = combined_result.get(
            "doctor",
            ""
        )

        if combined_doctor:

            y = draw_heading(
                "Combined Doctor Advice",
                y
            )

            y = draw_wrapped_text(
                combined_doctor,
                left_margin + 20,
                y,
                text_width - 20
            )

            y -= 20


    # ========================================================
    # ORIGINAL DISEASE INFORMATION
    # ========================================================

    y = draw_heading(
        "Disease Information",
        y
    )


    y = draw_wrapped_text(
        description,
        left_margin + 20,
        y,
        text_width - 20
    )

    y -= 15


    y = draw_heading(
        "Medicine / Treatment Information",
        y
    )


    y = draw_wrapped_text(
        medicine,
        left_margin + 20,
        y,
        text_width - 20
    )

    y -= 15


    y = draw_heading(
        "General Care / Home Remedy",
        y
    )


    y = draw_wrapped_text(
        remedy,
        left_margin + 20,
        y,
        text_width - 20
    )

    y -= 15


    y = draw_heading(
        "Doctor Advice",
        y
    )


    y = draw_wrapped_text(
        doctor,
        left_margin + 20,
        y,
        text_width - 20
    )

    y -= 25


    # ========================================================
    # DISCLAIMER
    # ========================================================

    y = draw_heading(
        "Important Medical Disclaimer",
        y
    )


    disclaimer = (
        "This report is generated by an Artificial Intelligence "
        "(AI) system and is intended for informational and "
        "educational purposes only. It is not a medical diagnosis "
        "and should not replace professional medical advice, "
        "clinical examination, laboratory testing, or treatment. "
        "Please consult a qualified dermatologist or healthcare "
        "professional for proper diagnosis and treatment."
    )


    y = draw_wrapped_text(
        disclaimer,
        left_margin,
        y,
        text_width,
        "Helvetica",
        9,
        14
    )


    # ========================================================
    # FOOTER
    # ========================================================

    c.setFont(
        "Helvetica",
        8
    )

    c.drawCentredString(
        page_width / 2,
        30,
        "Generated by DermaCare AI"
    )


    # ========================================================
    # SAVE PDF
    # ========================================================

    c.save()


    print(
        "PDF Created:",
        os.path.abspath(pdf_path)
    )


    return (
        pdf_path,
        pdf_filename
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(

        "index.html"

    )

# ============================================================
# ANIMAL CARE ROUTE
# ============================================================

@app.route("/animal-care")
def animal_care():

    # User logged in nahi hai
    if "user" not in session:

        return redirect(
            url_for(
                "login",
                type="animal"
            )
        )

    # User already logged in hai
    return redirect(
        url_for(
            "animal_dashboard"
        )
    )
    

# ============================================================

# LOGIN ROUTE

# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    # ========================================================
    # GET REQUEST
    # ========================================================

    if request.method == "GET":

        login_type = request.args.get(
          "type",
          "human"
        )

        return render_template(
            "login.html",
            login_type=login_type
        )


    # ========================================================
    # POST REQUEST
    # ========================================================

    email = request.form.get(
        "email",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    # Get hidden field from login.html
    login_type = request.form.get(
        "login_type",
        "human"
    ).strip().lower()

    # Safety check
    if login_type not in ["human", "animal"]:

        login_type = "human"


    print(
        "LOGIN TYPE:",
        login_type
    )


    # ========================================================
    # VALIDATE FORM
    # ========================================================

    if not email or not password:

        flash(
            "Please enter email and password.",
            "error"
        )

        return redirect(
            url_for(
                "login",
                type=login_type
            )
        )


    # ========================================================
    # DATABASE
    # ========================================================

    conn = get_db_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        )

        user = cursor.fetchone()

    finally:

        conn.close()


    # ========================================================
    # CHECK USER
    # ========================================================

    if user:

        try:

            password_valid = check_password_hash(
                user["password"],
                password
            )

        except Exception:

            password_valid = False


        # ====================================================
        # LOGIN SUCCESS
        # ====================================================

        if password_valid:

            # =================================================
            # SAVE USER SESSION
            # =================================================

            session["user"] = user["fullname"]

            session["user_id"] = user["id"]

            session["email"] = user["email"]

            # IMPORTANT:
            # Save dashboard type in session
            session["login_type"] = login_type


            print(
                "LOGIN SUCCESS:",
                user["email"]
            )

            print(
                "LOGIN TYPE:",
                login_type
            )


            # =================================================
            # ANIMAL DASHBOARD
            # =================================================

            if login_type == "animal":

                print(
                    "REDIRECTING TO ANIMAL DASHBOARD"
                )

                return redirect(
                    url_for(
                        "animal_dashboard"
                    )
                )


            # =================================================
            # HUMAN DASHBOARD
            # =================================================

            print(
                "REDIRECTING TO HUMAN DASHBOARD"
            )

            return redirect(
                url_for(
                    "dashboard"
                )
            )


    #=========================================================
    # INVALID LOGIN
    # ========================================================

    flash(
        "Invalid email or password.",
        "error"
    )

    return redirect(
        url_for(
            "login",
            type=login_type
        )
    )

   
# ============================================================
# ANIMAL DASHBOARD ROUTE
# ============================================================

@app.route("/animal-dashboard", methods=["GET", "POST"])
def animal_dashboard():

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if "user" not in session:

        return redirect(
            url_for(
                "login",
                type="animal"
            )
        )


    # ========================================================
    # USERNAME
    # ========================================================

    username = session["user"]
    
    # TEMPORARY POST HANDLING
    if request.method == "POST":

        animal_image = request.files.get("animal_image")

        if animal_image:

            print(
                "Animal Image:",
                animal_image.filename
            )

            # Abhi sirf testing ke liye
            # Next step me yahan AI model connect karenge


    # ========================================================
    # OPEN ANIMAL DASHBOARD
    # ========================================================

    return render_template(

        "animal_dashboard.html",

        username=username

    )


# ============================================================
# REGISTER ROUTE
# ============================================================

@app.route(

    "/register",

    methods=[

        "GET",

        "POST"

    ]

)

def register():

    if request.method == "POST":

        fullname = request.form.get(

            "fullname",

            ""

        ).strip()


        email = request.form.get(

            "email",

            ""

        ).strip()


        mobile = request.form.get(

            "mobile",

            ""

        ).strip()


        password = request.form.get(

            "password",

            ""

        )


        # ====================================================
        # VALIDATE INPUT
        # ====================================================

        if not fullname:

            flash(

                "Please enter your full name.",

                "error"

            )

            return redirect(

                url_for(
                    "register"
                )

            )


        if not email:

            flash(

                "Please enter your email.",

                "error"

            )

            return redirect(

                url_for(
                    "register"
                )

            )


        if not password:

            flash(

                "Please enter a password.",

                "error"

            )

            return redirect(

                url_for(
                    "register"
                )

            )


        # ====================================================
        # PASSWORD HASHING
        # ====================================================

        hashed_password = generate_password_hash(

            password

        )


        conn = get_db_connection()


        try:

            cursor = conn.cursor()


            # =================================================
            # CHECK EXISTING EMAIL
            # =================================================

            cursor.execute(

                """

                SELECT id

                FROM users

                WHERE email = ?

                """,

                (
                    email,
                )

            )


            existing_user = cursor.fetchone()


            if existing_user:

                flash(

                    "Email already exists. Please use another email.",

                    "error"

                )

                return redirect(

                    url_for(
                        "register"
                    )

                )


            # =================================================
            # INSERT USER
            # =================================================

            cursor.execute(

                """

                INSERT INTO users(

                    fullname,

                    email,

                    mobile,

                    password

                )

                VALUES(

                    ?,

                    ?,

                    ?,

                    ?

                )

                """,

                (

                    fullname,

                    email,

                    mobile,

                    hashed_password

                )

            )


            conn.commit()


            print(

                "REGISTRATION SUCCESS:",

                email

            )


        except Exception as e:

            conn.rollback()


            print(

                "Registration Error:",

                e

            )


            flash(

                "Registration failed. Please try again.",

                "error"

            )


            return redirect(

                url_for(
                    "register"
                )

            )


        finally:

            conn.close()


        flash(

            "Registration successful. Please login.",

            "success"

        )


        return redirect(

            url_for(
                "login"
            )

        )


    return render_template(

        "register.html"

    )


# ============================================================
# TEST ROUTE
# ============================================================

@app.route(

    "/test",

    methods=[

        "GET",

        "POST"

    ]

)

def test():

    return "POST Working"


# ============================================================
# USERS ROUTE
# ============================================================

# IMPORTANT:
# This route should ideally be protected by admin login.
# For college project testing, login is required.

@app.route(

    "/users"

)

def users():

    if "user" not in session:

        return redirect(

            url_for(
                "login"
            )

        )


    conn = get_db_connection()


    try:

        cursor = conn.cursor()


        cursor.execute(

            """

            SELECT

                id,

                fullname,

                email,

                mobile

            FROM users

            ORDER BY id DESC

            """

        )


        data = cursor.fetchall()


    finally:

        conn.close()


    return render_template(

        "users.html",

        users=data

    )

def analyze_skin_text(problem_text):

    print("\n" + "=" * 70)
    print("GEMINI TEXT ANALYSIS STARTED")
    print("=" * 70)

    prompt = f"""
You are an experienced dermatologist AI assistant.

User Problem:
{problem_text}

Return ONLY valid JSON.

{{
    "disease": "",
    "reason": "",
    "care": "",
    "medicine": "",
    "doctor": ""
}}

Important:
- Do not say it is a confirmed diagnosis.
- Recommend consulting a dermatologist.
- Keep the answer simple.
"""

    try:

        print("Calling Gemini API...")

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        print("Gemini API Response Received!")

        print("RAW RESPONSE:")
        print(response.text)

        import json

        result = json.loads(response.text)

        print("=" * 70)
        print("TEXT ANALYSIS COMPLETED")
        print(result)
        print("=" * 70)

        return result

    except Exception as e:

        print("=" * 70)
        print("Gemini Text Analysis Error:")
        print(type(e).__name__)
        print(e)
        print("=" * 70)

        return {
            "disease": "Analysis unavailable",
            "reason": "AI analysis could not be completed.",
            "care": "Please consult a dermatologist.",
            "medicine": "Do not self-medicate based only on this result.",
            "doctor": "Please consult a dermatologist for proper evaluation."
        }

# ============================================================
# DASHBOARD ROUTE
# ============================================================

@app.route(
    "/dashboard",
    methods=["GET", "POST"]
)
def dashboard():

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if "user" not in session:
        return redirect(
            url_for("login")
        )

    username = session["user"]

    # ========================================================
    # DATABASE STATISTICS
    # ========================================================

    total_users = 0
    total_predictions = 0
    total_reports = 0

    conn = get_db_connection()

    try:

        cursor = conn.cursor()

        # TOTAL USERS
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM users
            """
        )

        total_users = cursor.fetchone()[0]

        # TOTAL PREDICTIONS
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE username = ?
            """,
            (username,)
        )

        total_predictions = cursor.fetchone()[0]

        # TOTAL REPORTS
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM prediction_history
            WHERE username = ?
            AND pdf IS NOT NULL
            AND pdf != ''
            """,
            (username,)
        )

        total_reports = cursor.fetchone()[0]

    except Exception as e:

        print(
            "Dashboard Database Error:",
            e
        )

    finally:

        conn.close()

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    image_name = None

    prediction = None

    confidence = None

    description = None

    medicine = None

    remedy = None

    doctor = None

    pdf_filename = None

    text_data = None

    combined_result = None

    # ========================================================
    # POST REQUEST
    # ========================================================

    if request.method == "POST":

        # ====================================================
        # GET USER TEXT
        # ====================================================

        problem_text = request.form.get(
            "problem_text",
            ""
        ).strip()

        # ====================================================
        # GET IMAGE
        # ====================================================

        image = request.files.get(
            "image"
        )

        # ====================================================
        # TEXT ONLY ANALYSIS
        # ====================================================

        if image is None or image.filename == "":

            if problem_text:

                print(
                    "\n" + "=" * 70
                )

                print(
                    "TEXT ANALYSIS REQUEST RECEIVED"
                )

                print(
                    "User:",
                    username
                )

                print(
                    "Problem:",
                    problem_text
                )

                print(
                    "=" * 70
                )

                try:

                    text_data = analyze_skin_text(
                        problem_text
                    )

                except Exception as e:

                    print(
                        "Text Analysis Error:",
                        e
                    )

                    text_data = None

                return render_template(
                    "dashboard.html",

                    total_users=total_users,

                    total_predictions=total_predictions,

                    total_reports=total_reports,

                    image_name=None,

                    prediction=None,

                    confidence=None,

                    description=None,

                    medicine=None,

                    remedy=None,

                    doctor=None,

                    pdf_filename=None,

                    text_data=text_data,

                    combined_result=None
                )

            # NO IMAGE + NO TEXT

            flash(
                "Please select an image or enter your problem.",
                "error"
            )

            return render_template(
                "dashboard.html",

                total_users=total_users,

                total_predictions=total_predictions,

                total_reports=total_reports,

                image_name=None,

                prediction=None,

                confidence=None,

                description=None,

                medicine=None,

                remedy=None,

                doctor=None,

                pdf_filename=None,

                text_data=None,

                combined_result=None
            )

        # ====================================================
        # FILE VALIDATION
        # ====================================================

        if not allowed_file(image.filename):

            flash(
                "Invalid image format. Please upload JPG, JPEG, PNG, or WEBP.",
                "error"
            )

            return render_template(
                "dashboard.html",

                total_users=total_users,

                total_predictions=total_predictions,

                total_reports=total_reports,

                image_name=None,

                prediction="Invalid Image",

                confidence="0",

                description=disease_info["Invalid Image"]["description"],

                medicine=disease_info["Invalid Image"]["medicine"],

                remedy=disease_info["Invalid Image"]["remedy"],

                doctor=disease_info["Invalid Image"]["doctor"],

                pdf_filename=None,

                text_data=None,

                combined_result=None
            )

        # ====================================================
        # GENERATE UNIQUE FILENAME
        # ====================================================

        filename = generate_unique_filename(
            image.filename
        )

        image_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # ====================================================
        # SAVE IMAGE
        # ====================================================

        try:

            image.save(
                image_path
            )

            image_name = filename

            print(
                "\n" + "=" * 70
            )

            print(
                "IMAGE UPLOADED SUCCESSFULLY"
            )

            print(
                "Original Filename:",
                image.filename
            )

            print(
                "Saved Filename:",
                filename
            )

            print(
                "Saved Path:",
                os.path.abspath(image_path)
            )

            print(
                "=" * 70
            )

        except Exception as e:

            print(
                "Image Save Error:",
                e
            )

            flash(
                "Unable to save uploaded image.",
                "error"
            )

            return redirect(
                url_for("dashboard")
            )

        # ====================================================
        # STEP 1
        # SKIN IMAGE VALIDATION
        # ====================================================

        validator_result, validator_confidence = (
            validate_skin_image(
                image_path
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "VALIDATOR RESULT:",
            validator_result
        )

        print(
            "VALIDATOR CONFIDENCE:",
            validator_confidence,
            "%"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # STEP 2
        # DEFAULT DISEASE RESULT
        # ====================================================

        disease_prediction = "Invalid Image"

        disease_confidence = validator_confidence

        # ====================================================
        # RUN DISEASE MODEL
        # ====================================================

        if (
            validator_result == 1
            and
            validator_confidence >= VALIDATOR_CONFIDENCE_THRESHOLD
        ):

            disease_prediction, disease_confidence = (
                predict_human_disease(
                    image_path
                )
            )

        else:

            print(
                "Disease model skipped."
            )

            print(
                "Reason: Image failed skin validation."
            )

        # ====================================================
        # TEXT ANALYSIS
        # ====================================================

        text_data = None

        if problem_text:

            try:

                text_data = analyze_skin_text(
                    problem_text
                )

                print(
                    "\n" + "=" * 70
                )

                print(
                    "TEXT ANALYSIS COMPLETED"
                )

                print(
                    text_data
                )

                print(
                    "=" * 70
                )

            except Exception as e:

                print(
                    "Text Analysis Error:",
                    e
                )

                text_data = None

        # ====================================================
        # VALID DISEASE CLASSES
        # ====================================================

        valid_skin_diseases = [

            "Acne",

            "Eczema",

            "Healthy",

            "Melanoma",

            "Psoriasis",

            "Ringworm"

        ]

        # ====================================================
        # FINAL IMAGE VALIDATION
        # ====================================================

        final_valid = (

            validator_result == 1

            and

            validator_confidence >= VALIDATOR_CONFIDENCE_THRESHOLD

            and

            disease_prediction in valid_skin_diseases

            and

            disease_confidence >= DISEASE_CONFIDENCE_THRESHOLD

        )

        # ====================================================
        # DEBUG
        # ====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL VALIDATION DEBUG"
        )

        print(
            "Validator Result:",
            validator_result
        )

        print(
            "Validator Confidence:",
            validator_confidence,
            "%"
        )

        print(
            "Disease Prediction:",
            disease_prediction
        )

        print(
            "Disease Confidence:",
            disease_confidence,
            "%"
        )

        print(
            "Final Valid:",
            final_valid
        )

        print(
            "=" * 70
        )

        # ====================================================
        # INVALID IMAGE
        # ====================================================

        if not final_valid:

            prediction = "Invalid Image"

            confidence = round(
                float(disease_confidence),
                2
            )

            if validator_result == 0:

                confidence = round(
                    float(validator_confidence),
                    2
                )

            info = get_disease_information(
                "Invalid Image"
            )

            description = info["description"]

            medicine = info["medicine"]

            remedy = info["remedy"]

            doctor = info["doctor"]

            pdf_filename = None

            print(
                "FINAL RESULT: INVALID IMAGE"
            )

            print(
                "No PDF created."
            )

            print(
                "No prediction history saved."
            )

            flash(
                "Invalid image. Please upload a clear human skin image.",
                "error"
            )

        # ====================================================
        # VALID IMAGE
        # ====================================================

        else:

            # =================================================
            # INITIAL IMAGE RESULT
            # =================================================

            prediction = disease_prediction

            confidence = round(
                float(disease_confidence),
                2
            )

            # =================================================
            # COMBINED IMAGE + TEXT ANALYSIS
            # =================================================

            if problem_text and text_data:

                try:

                    combined_result = combine_analysis(

                        prediction,

                        confidence,

                        text_data

                    )

                    # =========================================
                    # COMBINED RESULT DEBUG
                    # =========================================

                    if combined_result:

                        combined_disease = (
                            combined_result.get(
                                "final_disease",
                                ""
                            )
                        )

                        match_level = (
                            combined_result.get(
                                "match_level",
                                ""
                            )
                        )

                        combined_confidence = (
                            combined_result.get(
                                "combined_confidence",
                                ""
                            )
                        )

                        print(
                            "\n" + "=" * 60
                        )

                        print(
                            "COMBINED RESULT APPLIED"
                        )

                        print(
                            "Image Prediction:",
                            prediction
                        )

                        print(
                            "Match Level:",
                            match_level
                        )

                        print(
                            "Combined Disease:",
                            combined_disease
                        )

                        print(
                            "Combined Confidence:",
                            combined_confidence
                        )

                        print(
                            "=" * 60
                        )

                        # =====================================
                        # MISMATCH / INCONCLUSIVE
                        # =====================================

                        if (
                            match_level == "MISMATCH"
                            or
                            combined_disease.strip().lower()
                            == "inconclusive"
                        ):

                            prediction = "Inconclusive"

                            # Keep confidence as clean string
                            confidence = str(
                                combined_confidence
                            ).replace(
                                "%",
                                ""
                            ).strip()

                            if not confidence:

                                confidence = "Low"

                            # ---------------------------------
                            # IMPORTANT:
                            # Do NOT call get_disease_information()
                            # here because Inconclusive is not a
                            # disease class.
                            # ---------------------------------

                            description = (
                                "The image prediction and reported "
                                "symptoms do not sufficiently match. "
                                "The result is inconclusive and should "
                                "not be treated as a confirmed diagnosis."
                            )

                            medicine = combined_result.get(
                                "medicine",
                                "No disease-specific treatment should "
                                "be assumed from conflicting results."
                            )

                            remedy = combined_result.get(
                                "care",
                                "Follow general skin care and seek "
                                "professional medical evaluation."
                            )

                            doctor = combined_result.get(
                                "doctor",
                                "Please consult a qualified dermatologist "
                                "or healthcare professional."
                            )

                        # =====================================
                        # MATCH
                        # =====================================

                        elif match_level == "MATCH":

                            prediction = (
                                combined_disease
                                if combined_disease
                                else disease_prediction
                            )

                            confidence = str(
                                combined_confidence
                            ).replace(
                                "%",
                                ""
                            ).strip()

                            # =================================================
                            # GET FINAL DISEASE INFORMATION
                            # =================================================
                            #
                            # IMPORTANT:
                            # If Combined AI Analysis has already produced the
                            # final result, DO NOT overwrite it using
                            # get_disease_information().
                            #
                            # This is especially important for:
                            # MATCH       -> keep combined AI result
                            # MISMATCH    -> keep Inconclusive result
                            # UNCERTAIN   -> keep Inconclusive result
                            # =================================================

                            if combined_result:

                                match_level = str(
                                    combined_result.get(
                                        "match_level",
                                        ""
                                    )
                                ).upper()

                                combined_disease = str(
                                    combined_result.get(
                                        "final_disease",
                                        ""
                                    )
                                ).strip()


                                # =================================================
                                # MISMATCH / INCONCLUSIVE
                                # =================================================

                                if (
                                    match_level == "MISMATCH"
                                    or
                                    combined_disease.lower() == "inconclusive"
                                ):

                                    prediction = "Inconclusive"


                                    # -------------------------------------------------
                                    # Combined confidence
                                    # -------------------------------------------------

                                    combined_confidence = combined_result.get(
                                        "combined_confidence",
                                        ""
                                    )


                                    confidence = combined_confidence


                                    # -------------------------------------------------
                                    # Description
                                    # -------------------------------------------------

                                    description = combined_result.get(
                                        "assessment",
                                        "The image prediction and reported symptoms "
                                        "do not sufficiently match. The result is "
                                        "inconclusive and should not be treated as "
                                        "a confirmed diagnosis."
                                    )


                                    # -------------------------------------------------
                                    # Medicine
                                    # -------------------------------------------------

                                    medicine = combined_result.get(
                                        "medicine",
                                        "No disease-specific treatment should be "
                                        "assumed from the conflicting results."
                                    )


                                    # -------------------------------------------------
                                    # Care / Remedy
                                    # -------------------------------------------------

                                    remedy = combined_result.get(
                                        "care",
                                        "Follow general safe skin care and seek "
                                        "professional medical evaluation."
                                    )


                                    # -------------------------------------------------
                                    # Doctor
                                    # -------------------------------------------------

                                    doctor = combined_result.get(
                                        "doctor",
                                        "Please consult a qualified dermatologist "
                                        "or healthcare professional."
                                    )


                                # =================================================
                                # MATCH
                                # =================================================

                                elif match_level == "MATCH":

                                    prediction = combined_disease


                                    # -------------------------------------------------
                                    # Combined confidence
                                    # -------------------------------------------------

                                    confidence = combined_result.get(
                                        "combined_confidence",
                                        confidence
                                    )


                                    # -------------------------------------------------
                                    # Get basic disease information
                                    # -------------------------------------------------

                                    info = get_disease_information(
                                        prediction
                                    )


                                    # -------------------------------------------------
                                    # Description
                                    # -------------------------------------------------

                                    description = info.get(
                                        "description",
                                        ""
                                    )


                                    # -------------------------------------------------
                                    # Combined AI recommendations
                                    # take priority over database information
                                    # -------------------------------------------------

                                    medicine = combined_result.get(
                                        "medicine"
                                    ) or info.get(
                                        "medicine",
                                        ""
                                    )


                                    remedy = combined_result.get(
                                        "care"
                                    ) or info.get(
                                        "remedy",
                                        ""
                                    )


                                    doctor = combined_result.get(
                                        "doctor"
                                    ) or info.get(
                                        "doctor",
                                        ""
                                    )


                                # =================================================
                                # UNCERTAIN
                                # =================================================

                                else:

                                    prediction = "Inconclusive"


                                    confidence = combined_result.get(
                                        "combined_confidence",
                                        "0%"
                                    )


                                    description = combined_result.get(
                                        "assessment",
                                        "The available image and symptom information "
                                        "is insufficient for a reliable assessment."
                                    )


                                    medicine = combined_result.get(
                                        "medicine",
                                        "Please consult a qualified healthcare "
                                        "professional before using medication."
                                    )


                                    remedy = combined_result.get(
                                        "care",
                                        "Follow general gentle skin care and seek "
                                        "professional evaluation."
                                    )


                                    doctor = combined_result.get(
                                        "doctor",
                                        "Please consult a qualified dermatologist "
                                        "or healthcare professional."
                                    )


                            else:

                                # =================================================
                                # NO COMBINED RESULT
                                # =================================================
                                # Normal image-only prediction
                                # =================================================

                                info = get_disease_information(
                                    prediction
                                )


                                description = info.get(
                                    "description",
                                    ""
                                )


                                medicine = info.get(
                                    "medicine",
                                    ""
                                )


                                remedy = info.get(
                                    "remedy",
                                    ""
                                )


                                doctor = info.get(
                                    "doctor",
                                    ""
                                )

                        # =====================================
                        # UNCERTAIN
                        # =====================================

                        else:

                            prediction = "Inconclusive"

                            confidence = str(
                                combined_confidence
                            ).replace(
                                "%",
                                ""
                            ).strip()

                            if not confidence:

                                confidence = "Low"

                            description = (
                                "The available image and symptom "
                                "information is insufficient for a "
                                "reliable combined assessment."
                            )

                            medicine = combined_result.get(
                                "medicine",
                                "Please consult a qualified healthcare professional."
                            )

                            remedy = combined_result.get(
                                "care",
                                "Follow gentle general skin care."
                            )

                            doctor = combined_result.get(
                                "doctor",
                                "Please consult a qualified dermatologist "
                                "or healthcare professional."
                            )

                    print(
                        "\n" + "=" * 70
                    )

                    print(
                        "COMBINED AI ANALYSIS"
                    )

                    print(
                        combined_result
                    )

                    print(
                        "=" * 70
                    )

                except Exception as e:

                    print(
                        "Combined Analysis Error:",
                        e
                    )

                    combined_result = None

            # =================================================
            # NO TEXT / COMBINED ANALYSIS FAILED
            # =================================================

            if not problem_text or not combined_result:

                info = get_disease_information(
                    prediction
                )

                description = info[
                    "description"
                ]

                medicine = info[
                    "medicine"
                ]

                remedy = info[
                    "remedy"
                ]

                doctor = info[
                    "doctor"
                ]

            # =================================================
            # CREATE PDF
            # =================================================

            try:

                pdf_path, pdf_filename = create_pdf(

                    username,

                    prediction,

                    confidence,

                    description,

                    medicine,

                    remedy,

                    doctor,

                    problem_text,

                    combined_result

                )

            except Exception as e:

                print(
                    "PDF Creation Error:",
                    e
                )

                pdf_filename = None

            # =================================================
            # SAVE PREDICTION HISTORY
            # =================================================

            try:

                conn = get_db_connection()

                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO prediction_history(

                        username,

                        category,

                        disease,

                        confidence,

                        image,

                        pdf

                    )

                    VALUES(

                        ?,
                        ?,
                        ?,
                        ?,
                        ?,
                        ?

                    )
                    """,

                    (

                        username,

                        "human",

                        prediction,

                        confidence,

                        image_name,

                        pdf_filename

                    )

                )

                conn.commit()

                print(
                    "Prediction History Saved Successfully!"
                )

            except Exception as e:

                print(
                    "Prediction History Error:",
                    e
                )

            finally:

                try:

                    conn.close()

                except Exception:

                    pass

            # =================================================
            # FINAL DEBUG
            # =================================================

            print(
                "FINAL RESULT:",
                prediction
            )

            print(
                "FINAL CONFIDENCE:",
                confidence
            )

            flash(
                f"Prediction completed: {prediction}",
                "success"
            )

    # ========================================================
    # RENDER DASHBOARD
    # ========================================================

    return render_template(

        "dashboard.html",

        total_users=total_users,

        total_predictions=total_predictions,

        total_reports=total_reports,

        image_name=image_name,

        prediction=prediction,

        confidence=confidence,

        description=description,

        medicine=medicine,

        remedy=remedy,

        doctor=doctor,

        pdf_filename=pdf_filename,

        text_data=text_data,

        combined_result=combined_result

    )



# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route(

    "/download_report/<filename>"

)

def download_report(

    filename

):

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if "user" not in session:

        return redirect(

            url_for(
                "login"
            )

        )


    # ========================================================
    # SECURITY
    # ========================================================

    filename = secure_filename(

        filename

    )


    # ========================================================
    # CHECK USER OWNS PDF
    # ========================================================

    conn = get_db_connection()


    try:

        cursor = conn.cursor()


        cursor.execute(

            """

            SELECT id

            FROM prediction_history

            WHERE username = ?

            AND pdf = ?

            """,

            (

                session["user"],

                filename

            )

        )


        record = cursor.fetchone()


    finally:

        conn.close()


    if not record:

        return (

            "Unauthorized or report not found",

            403

        )


    # ========================================================
    # SEND PDF
    # ========================================================

    return send_from_directory(

        PDF_FOLDER,

        filename,

        as_attachment=True

    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    print(
        "LOGOUT:",
        session.get("user")
    )

    # ========================================================
    # GET LOGOUT DESTINATION
    # ========================================================

    next_page = request.args.get(
        "next",
        "human"
    )


    # ========================================================
    # CLEAR SESSION
    # ========================================================

    session.clear()


    # ========================================================
    # REDIRECT TO CORRECT LOGIN
    # ========================================================

    if next_page == "animal":

        print(
            "LOGOUT REDIRECT: ANIMAL LOGIN"
        )

        return redirect(
            url_for(
                "login",
                type="animal"
            )
        )


    print(
        "LOGOUT REDIRECT: HUMAN LOGIN"
    )

    return redirect(
        url_for(
            "login",
            type="human"
        )
    )


# ============================================================
# HISTORY
# ============================================================

@app.route(

    "/history"

)

def history():

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if "user" not in session:

        return redirect(

            url_for(
                "login"
            )

        )


    username = session["user"]


    # ========================================================
    # SEARCH
    # ========================================================

    search = request.args.get(

        "search",

        ""

    ).strip()


    # ========================================================
    # DATABASE
    # ========================================================

    conn = get_db_connection()


    try:

        cursor = conn.cursor()


        if search:

            cursor.execute(

                """

                SELECT

                    id,

                    disease,

                    confidence,

                    image,

                    prediction_date,

                    pdf

                FROM prediction_history

                WHERE username = ?
                
                AND category = 'human'

                AND disease LIKE ?

                ORDER BY prediction_date DESC

                """,

                (

                    username,

                    "%" + search + "%"

                )

            )


        else:

            cursor.execute(

                """

                SELECT

                    id,

                    disease,

                    confidence,

                    image,

                    prediction_date,

                    pdf

                FROM prediction_history

                WHERE username = ?
                
                AND category = 'human'

                ORDER BY prediction_date DESC

                """,

                (

                    username,

                )

            )


        history_data = cursor.fetchall()


    finally:

        conn.close()


    return render_template(

        "history.html",

        history_data=history_data,

        search=search

    )


# ============================================================
# DELETE HISTORY
# ============================================================

@app.route(

    "/delete_history/<int:history_id>",

    methods=[

        "POST"

    ]

)

def delete_history(

    history_id

):

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if "user" not in session:

        return redirect(

            url_for(
                "login"
            )

        )


    username = session["user"]


    # ========================================================
    # DATABASE
    # ========================================================

    conn = get_db_connection()


    try:

        cursor = conn.cursor()


        # ====================================================
        # GET CURRENT USER RECORD
        # ====================================================

        cursor.execute(

            """

            SELECT

                image,

                pdf

            FROM prediction_history

            WHERE id = ?

            AND username = ?

            """,

            (

                history_id,

                username

            )

        )


        record = cursor.fetchone()


        # ====================================================
        # RECORD NOT FOUND
        # ====================================================

        if not record:

            conn.close()


            flash(

                "History record not found.",

                "error"

            )


            return redirect(

                url_for(
                    "history"
                )

            )


        image_name = record["image"]

        pdf_name = record["pdf"]


        # ====================================================
        # DELETE DATABASE RECORD
        # ====================================================

        cursor.execute(

            """

            DELETE FROM prediction_history

            WHERE id = ?

            AND username = ?

            """,

            (

                history_id,

                username

            )

        )


        conn.commit()


    except Exception as e:

        conn.rollback()


        print(

            "Delete History Error:",

            e

        )


        flash(

            "Unable to delete history record.",

            "error"

        )


        return redirect(

            url_for(
                "history"
            )

        )


    finally:

        conn.close()


    # ========================================================
    # DELETE IMAGE
    # ========================================================

    if image_name:

        image_name = secure_filename(

            image_name

        )


        image_path = os.path.join(

            UPLOAD_FOLDER,

            image_name

        )


        if os.path.exists(

            image_path

        ):

            try:

                os.remove(

                    image_path

                )

                print(

                    "Deleted Image:",

                    image_path

                )

            except Exception as e:

                print(

                    "Image Delete Error:",

                    e

                )


    # ========================================================
    # DELETE PDF
    # ========================================================

    if pdf_name:

        pdf_name = secure_filename(

            pdf_name

        )


        pdf_path = os.path.join(

            PDF_FOLDER,

            pdf_name

        )


        if os.path.exists(

            pdf_path

        ):

            try:

                os.remove(

                    pdf_path

                )

                print(

                    "Deleted PDF:",

                    pdf_path

                )

            except Exception as e:

                print(

                    "PDF Delete Error:",

                    e

                )


    flash(

        "History deleted successfully.",

        "success"

    )


    return redirect(

        url_for(
            "history"
        )

    )


# ============================================================
# ERROR HANDLER - FILE TOO LARGE
# ============================================================

@app.errorhandler(

    413

)

def file_too_large(

    error

):

    flash(

        "File is too large. Maximum allowed size is 10 MB.",

        "error"

    )


    return redirect(

        url_for(
            "dashboard"
        )

    )


# ============================================================
# ERROR HANDLER - GENERAL
# ============================================================

@app.errorhandler(

    500

)

def internal_server_error(

    error

):

    print(

        "Internal Server Error:",

        error

    )


    return (

        "Internal Server Error. Please check the Flask terminal.",

        500

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "DERMACARE AI SERVER STARTING..."
    )

    print(
        "Open on PC:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "For Mobile:"
    )

    print(
        "Use your PC IPv4 Address + :5000"
    )

    print("=" * 70)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )