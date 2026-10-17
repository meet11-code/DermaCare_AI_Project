import os
import sqlite3
import tensorflow as tf
import numpy as np

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash


# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "dermacare_ai_secret_key"

# Upload folder
UPLOAD_FOLDER = "static/uploads"

# PDF report folder
PDF_FOLDER = "static/reports"

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# STARTING INFORMATION
# ============================================================

print("=" * 60)
print("DERMACARE AI STARTING")
print("=" * 60)

print("Project Directory:")
print(os.path.abspath("."))

print("Upload Directory:")
print(os.path.abspath(UPLOAD_FOLDER))

print("Reports Directory:")
print(os.path.abspath(PDF_FOLDER))

print("Database Path:")
print(os.path.abspath("users.db"))


# ============================================================
# MODEL PATHS
# ============================================================

SKIN_MODEL_PATH = os.path.join(
    "model",
    "skin_model_v2.keras"
)

VALIDATOR_MODEL_PATH = os.path.join(
    "model",
    "validator_model.keras"
)


# ============================================================
# LOAD SKIN DISEASE MODEL
# ============================================================

print("\n" + "=" * 60)
print("Loading Skin Disease Model...")
print("=" * 60)

if os.path.exists(SKIN_MODEL_PATH):

    print("Model Found:")
    print(os.path.abspath(SKIN_MODEL_PATH))

    model = tf.keras.models.load_model(
        SKIN_MODEL_PATH
    )

    print("Skin Disease Model Loaded Successfully!")

else:

    print("WARNING: Skin Disease Model Not Found!")

    model = None


# ============================================================
# LOAD VALIDATOR MODEL
# ============================================================

print("\n" + "=" * 60)
print("Loading Skin Validator Model...")
print("=" * 60)

if os.path.exists(VALIDATOR_MODEL_PATH):

    print("Validator Model Found:")
    print(os.path.abspath(VALIDATOR_MODEL_PATH))

    validator_model = tf.keras.models.load_model(
        VALIDATOR_MODEL_PATH
    )

    print("Validator Model Loaded Successfully!")

else:

    print("WARNING: Validator Model Not Found!")

    validator_model = None


# ============================================================
# HUMAN DISEASE CLASS ORDER
# IMPORTANT:
# THIS MUST MATCH TRAINING CLASS ORDER
# ============================================================

human_classes = [

    "Acne",

    "Eczema",

    "Healthy",

    "Melanoma",

    "Psoriasis",

    "Ringworm"

]


print("\n" + "=" * 60)
print("HUMAN DISEASE CLASSES")
print("=" * 60)

for index, class_name in enumerate(human_classes):

    print(
        f"{index} -> {class_name}"
    )

print(
    "Total Classes:",
    len(human_classes)
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    "Acne": {

        "description":
        "Acne is a common skin disease caused by clogged pores.",

        "medicine":
        "Benzoyl Peroxide Gel may be used for some acne conditions. Consult a healthcare professional for appropriate treatment.",

        "remedy":
        "Wash the affected area gently and avoid picking or squeezing pimples.",

        "doctor":
        "Consult a dermatologist if acne is severe or persistent."

    },


    "Eczema": {

        "description":
        "Eczema causes dry, itchy and inflamed skin.",

        "medicine":
        "Moisturizer may help. Medical treatment should be taken according to a doctor's advice.",

        "remedy":
        "Use a gentle moisturizer and avoid harsh soaps or known irritants.",

        "doctor":
        "Consult a dermatologist if itching, redness or inflammation becomes severe."

    },


    "Healthy": {

        "description":
        "The uploaded image appears to show healthy skin with no major visible skin disease detected by the AI model.",

        "medicine":
        "No medicine is required based on this AI prediction.",

        "remedy":
        "Maintain regular skin hygiene, use a suitable moisturizer and protect skin from excessive sunlight.",

        "doctor":
        "No immediate consultation is indicated by this prediction. Consult a dermatologist if you notice unusual or persistent skin changes."

    },


    "Melanoma": {

        "description":
        "The AI model detected a pattern that may be associated with melanoma. This result is not a medical diagnosis.",

        "medicine":
        "Do not self-medicate. Professional medical evaluation is recommended.",

        "remedy":
        "Avoid excessive sun exposure and do not delay professional medical evaluation.",

        "doctor":
        "Consult a dermatologist or qualified medical professional as soon as possible for proper examination."

    },


    "Psoriasis": {

        "description":
        "Psoriasis can cause thick, red, scaly or inflamed patches on the skin.",

        "medicine":
        "Treatment depends on severity and should be recommended by a healthcare professional.",

        "remedy":
        "Keep the skin moisturized and avoid known triggers or irritants.",

        "doctor":
        "Consult a dermatologist for proper diagnosis and treatment."

    },


    "Ringworm": {

        "description":
        "Ringworm is a fungal skin infection that may cause an itchy, circular rash.",

        "medicine":
        "Antifungal treatment may be required. Consult a healthcare professional for appropriate treatment.",

        "remedy":
        "Keep the affected area clean and dry and avoid sharing personal items.",

        "doctor":
        "Consult a dermatologist if the infection spreads or does not improve."

    }

}


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    print("\nPreprocessing Image...")

    # Open image
    img = Image.open(
        image_path
    ).convert("RGB")

    # Resize
    img = img.resize(
        (224, 224)
    )

    # Convert to NumPy
    img_array = np.array(
        img,
        dtype=np.float32
    )

    # IMPORTANT:
    # Normalize pixel values from 0-255 to 0-1
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# VALIDATE SKIN IMAGE
# ============================================================

def validate_skin_image(image_path):

    print("\n" + "=" * 60)
    print("VALIDATING UPLOADED IMAGE")
    print("=" * 60)

    print(
        "Image Path:"
    )

    print(
        os.path.abspath(image_path)
    )


    # Check image exists
    if not os.path.exists(image_path):

        print(
            "ERROR: Image file does not exist!"
        )

        return 0, 0.0


    # Validator not loaded
    if validator_model is None:

        print(
            "WARNING: Validator model not loaded."
        )

        print(
            "Assuming image is valid."
        )

        return 1, 100.0


    try:

        # Preprocess image
        img_array = preprocess_image(
            image_path
        )


        # Validator prediction
        prediction = validator_model.predict(
            img_array,
            verbose=0
        )


        # Get class
        predicted_class = int(
            np.argmax(prediction)
        )


        # Get confidence
        confidence = float(
            np.max(prediction) * 100
        )


        print(
            "Validator Prediction:",
            prediction
        )

        print(
            "Validator Class:",
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


        # Validator class order
        #
        # 0 = non_skin
        # 1 = skin


        if predicted_class == 0:

            print(
                "Validator Result: NON-SKIN"
            )

        else:

            print(
                "Validator Result: SKIN"
            )


        return (
            predicted_class,
            confidence
        )


    except Exception as e:

        print(
            "Validator Error:",
            e
        )

        return (
            0,
            0.0
        )


# ============================================================
# PREDICT HUMAN SKIN DISEASE
# ============================================================

def predict_human_disease(image_path):

    print("\n" + "=" * 60)
    print("PREDICTING HUMAN SKIN DISEASE")
    print("=" * 60)


    if model is None:

        print(
            "ERROR: Skin disease model not loaded!"
        )

        return (
            "Model not loaded",
            0.0
        )


    try:

        # Preprocess image
        img_array = preprocess_image(
            image_path
        )


        # Disease prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )


        print(
            "Disease Prediction Array:",
            prediction
        )


        # Predicted class
        predicted_index = int(
            np.argmax(prediction)
        )


        # Confidence
        confidence = float(
            np.max(prediction) * 100
        )


        # Safety check
        if predicted_index >= len(
            human_classes
        ):

            print(
                "ERROR: Predicted class index is outside human_classes!"
            )

            return (
                "Prediction Error",
                0.0
            )


        # Disease name
        disease = human_classes[
            predicted_index
        ]


        print(
            "Predicted Class:",
            predicted_index
        )

        print(
            "Predicted Disease:",
            disease
        )

        print(
            "Disease Confidence:",
            round(
                confidence,
                2
            ),
            "%"
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
# CREATE PDF REPORT
# ============================================================

def create_pdf(
    username,
    disease,
    confidence,
    description,
    medicine,
    remedy,
    doctor
):

    safe_username = secure_filename(
        str(username)
    )


    pdf_path = os.path.join(

        PDF_FOLDER,

        f"medical_report_{safe_username}.pdf"

    )


    c = canvas.Canvas(

        pdf_path,

        pagesize=letter

    )


    # Title
    c.setFont(
        "Helvetica-Bold",
        18
    )

    c.drawString(
        180,
        760,
        "DermaCare AI"
    )


    # User
    c.setFont(
        "Helvetica",
        12
    )

    c.drawString(
        50,
        720,
        f"User : {username}"
    )


    # Disease
    c.drawString(
        50,
        690,
        f"Disease : {disease}"
    )


    # Confidence
    c.drawString(
        50,
        660,
        f"Confidence : {confidence}%"
    )


    # Description
    c.drawString(
        50,
        630,
        "Description :"
    )

    c.drawString(
        70,
        610,
        description[:100]
    )


    # Medicine
    c.drawString(
        50,
        580,
        "Medicine :"
    )

    c.drawString(
        70,
        560,
        medicine[:100]
    )


    # Home Remedy
    c.drawString(
        50,
        530,
        "Home Remedy :"
    )

    c.drawString(
        70,
        510,
        remedy[:100]
    )


    # Doctor Advice
    c.drawString(
        50,
        480,
        "Doctor Advice :"
    )

    c.drawString(
        70,
        460,
        doctor[:100]
    )


    # Save
    c.save()


    print(
        "PDF Created:",
        os.path.abspath(pdf_path)
    )


    return pdf_path


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form[
            "email"
        ]

        password = request.form[
            "password"
        ]


        conn = sqlite3.connect(
            "users.db"
        )

        cursor = conn.cursor()


        cursor.execute(

            "SELECT * FROM users WHERE email=?",

            (email,)

        )


        user = cursor.fetchone()


        conn.close()


        if user and check_password_hash(

            user[4],

            password

        ):

            session[
                "user"
            ] = user[1]


            return redirect(
                "/dashboard"
            )


        else:

            return (
                "<h1>❌ Invalid Email or Password</h1>"
            )


    return render_template(
        "login.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        fullname = request.form[
            "fullname"
        ]

        email = request.form[
            "email"
        ]

        mobile = request.form[
            "mobile"
        ]

        password = request.form[
            "password"
        ]


        hashed_password = generate_password_hash(
            password
        )


        conn = sqlite3.connect(
            "users.db"
        )

        cursor = conn.cursor()


        try:

            cursor.execute(

                """
                INSERT INTO users(
                    fullname,
                    email,
                    mobile,
                    password
                )
                VALUES(?, ?, ?, ?)
                """,

                (
                    fullname,
                    email,
                    mobile,
                    hashed_password
                )

            )


            conn.commit()

            conn.close()


            return redirect(
                "/login"
            )


        except Exception as e:

            conn.close()


            return (
                f"Error: {e}"
            )


    return render_template(
        "register.html"
    )


# ============================================================
# TEST ROUTE
# ============================================================

@app.route(
    "/test",
    methods=["POST"]
)
def test():

    return "POST Working"


# ============================================================
# USERS
# ============================================================

@app.route(
    "/users"
)
def users():

    conn = sqlite3.connect(
        "users.db"
    )

    cursor = conn.cursor()


    cursor.execute(
        "SELECT * FROM users"
    )


    data = cursor.fetchall()


    conn.close()


    return str(
        data
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route(
    "/dashboard",
    methods=["GET", "POST"]
)
def dashboard():


    # Check login
    if "user" not in session:

        return redirect(
            "/login"
        )


    # ========================================================
    # TOTAL USERS
    # ========================================================

    conn = sqlite3.connect(
        "users.db"
    )

    cursor = conn.cursor()


    cursor.execute(
        "SELECT COUNT(*) FROM users"
    )


    total_users = cursor.fetchone()[0]


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


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    if request.method == "POST":


        image = request.files.get(
            "image"
        )


        if image and image.filename != "":


            # Secure filename
            filename = secure_filename(
                image.filename
            )


            # Image path
            image_path = os.path.join(

                app.config[
                    "UPLOAD_FOLDER"
                ],

                filename

            )


            # Save image
            image.save(
                image_path
            )


            print("\n" + "=" * 60)

            print(
                "IMAGE UPLOADED SUCCESSFULLY"
            )

            print(
                "Filename:",
                filename
            )

            print(
                "Saved Path:",
                os.path.abspath(
                    image_path
                )
            )

            print("=" * 60)


            # =================================================
            # STEP 1
            # VALIDATOR MODEL
            # =================================================

            validator_result, validator_confidence = (

                validate_skin_image(
                    image_path
                )

            )


            # =================================================
            # STEP 2
            # RUN DISEASE MODEL
            # =================================================

            # We run disease prediction even if
            # validator says NON-SKIN.
            #
            # This helps handle cases where
            # Healthy skin is incorrectly rejected
            # by the validator model.

            disease_prediction, disease_confidence = (

                predict_human_disease(
                    image_path
                )

            )


            # =================================================
            # STEP 3
            # FINAL VALIDATION LOGIC
            # =================================================

            # If validator says NON-SKIN,
            # but disease model predicts Healthy,
            # we accept the image as valid.

            if (

                validator_result == 0

                and disease_prediction == "Healthy"

            ):

                print(
                    "Validator said NON-SKIN,"
                )

                print(
                    "but Disease Model predicted HEALTHY."
                )

                print(
                    "FINAL DECISION: ACCEPT IMAGE"
                )

                final_valid = True


            elif validator_result == 1:

                print(
                    "Validator Result: SKIN"
                )

                print(
                    "FINAL DECISION: ACCEPT IMAGE"
                )

                final_valid = True


            else:

                print(
                    "FINAL DECISION: INVALID IMAGE"
                )

                final_valid = False


            # =================================================
            # STEP 4
            # INVALID IMAGE
            # =================================================

            if not final_valid:


                prediction = (
                    "Invalid Image"
                )


                confidence = round(

                    validator_confidence,

                    2

                )


                description = (

                    "The uploaded image could not be confidently "
                    "identified as a suitable human skin image. "
                    "Please upload a clear close-up image of "
                    "the affected skin area."

                )


                medicine = "-"


                remedy = "-"


                doctor = (

                    "Please upload a clear human skin image."

                )


                image_name = filename


                print(
                    "FINAL RESULT: INVALID IMAGE"
                )


            # =================================================
            # STEP 5
            # VALID IMAGE
            # =================================================

            else:


                # Use disease model prediction
                prediction = disease_prediction

                confidence = disease_confidence


                # =================================================
                # GET DISEASE INFORMATION
                # =================================================

                if prediction in disease_info:


                    info = disease_info[
                        prediction
                    ]


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


                else:


                    description = (

                        "No information available "
                        "for this prediction."

                    )


                    medicine = (

                        "Consult a healthcare professional."

                    )


                    remedy = (

                        "Consult a healthcare professional."

                    )


                    doctor = (

                        "Consult a dermatologist."

                    )


                # =================================================
                # CREATE PDF
                # =================================================

                create_pdf(

                    session[
                        "user"
                    ],

                    prediction,

                    confidence,

                    description,

                    medicine,

                    remedy,

                    doctor

                )


                # =================================================
                # SAVE HISTORY
                # =================================================

                image_name = filename


                conn = sqlite3.connect(
                    "users.db"
                )

                cursor = conn.cursor()


                cursor.execute(

                    """
                    INSERT INTO prediction_history(
                        username,
                        disease,
                        confidence,
                        image
                    )
                    VALUES (?, ?, ?, ?)
                    """,

                    (

                        session[
                            "user"
                        ],

                        prediction,

                        confidence,

                        image_name

                    )

                )


                conn.commit()

                conn.close()


                print(
                    "Prediction History Saved Successfully!"
                )


                print(
                    "FINAL RESULT:",
                    prediction
                )

                print(
                    "FINAL CONFIDENCE:",
                    confidence,
                    "%"
                )


    # ========================================================
    # RENDER DASHBOARD
    # ========================================================

    return render_template(

        "dashboard.html",

        total_users=total_users,

        image_name=image_name,

        prediction=prediction,

        confidence=confidence,

        description=description,

        medicine=medicine,

        remedy=remedy,

        doctor=doctor

    )


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history"
)
def history():


    if "user" not in session:

        return redirect(
            "/login"
        )


    search = request.args.get(
        "search",
        ""
    )


    conn = sqlite3.connect(
        "users.db"
    )

    cursor = conn.cursor()


    cursor.execute(

        """
        SELECT
            id,
            disease,
            confidence,
            image,
            prediction_date
        FROM prediction_history
        WHERE username=?
        AND disease LIKE ?
        ORDER BY id DESC
        """,

        (

            session[
                "user"
            ],

            "%" + search + "%"

        )

    )


    history_data = cursor.fetchall()


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
    "/delete_history/<int:id>"
)
def delete_history(id):


    if "user" not in session:

        return redirect(
            "/login"
        )


    conn = sqlite3.connect(
        "users.db"
    )

    cursor = conn.cursor()


    cursor.execute(

        "DELETE FROM prediction_history WHERE id=?",

        (id,)

    )


    conn.commit()

    conn.close()


    return redirect(
        "/history"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "DERMACARE AI SERVER STARTING..."
    )

    print("=" * 60)


    app.run(
        debug=True
    )