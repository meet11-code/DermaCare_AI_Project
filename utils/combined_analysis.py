import json


# ============================================================
# COMBINED IMAGE + TEXT ANALYSIS
# ============================================================

def combine_analysis(
    image_prediction,
    confidence,
    text_data
):

    # ========================================================
    # TEXT DATA SAFETY
    # ========================================================

    if not text_data:
        text_data = {}

    text_disease = text_data.get(
        "disease",
        ""
    )

    text_reason = text_data.get(
        "reason",
        ""
    )

    text_care = text_data.get(
        "care",
        ""
    )

    text_medicine = text_data.get(
        "medicine",
        ""
    )

    text_doctor = text_data.get(
        "doctor",
        ""
    )

    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    from app import client

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are an AI skin-analysis assistant.

Your task is to compare an image-model prediction with
the user's reported symptoms.

IMPORTANT:
The image model prediction is NOT a confirmed diagnosis.

IMAGE MODEL RESULT:

Predicted Disease:
{image_prediction}

Image Confidence:
{confidence}%

USER REPORTED SYMPTOMS:

Possible Disease:
{text_disease}

Reason / Symptoms:
{text_reason}

General Care:
{text_care}

Medicine Information:
{text_medicine}

Doctor Advice:
{text_doctor}


============================================================
IMPORTANT COMPARISON RULES
============================================================

1. Compare the image-model prediction with the user's
reported symptoms.

2. BODY LOCATION IS VERY IMPORTANT.

If the user reports a specific body location such as:

- hands
- face
- scalp
- legs
- feet
- arms
- chest
- back

you must consider that location when deciding MATCH,
MISMATCH, or UNCERTAIN.

3. Examples:

Acne commonly affects:
- face
- forehead
- cheeks
- chin
- chest
- shoulders
- back

Acne symptoms may include:
- pimples
- blackheads
- whiteheads
- papules
- pustules
- oily skin
- clogged pores

Hand eczema/contact dermatitis commonly affects:
- hands
- fingers
- palms
- backs of hands

Typical symptoms include:
- itching
- redness
- dry skin
- patches
- irritation
- cracking
- burning
- swelling

4. VERY IMPORTANT:

Do NOT automatically treat a disease prediction as a
MATCH only because the disease name appears somewhere
inside the text analysis.

For example:

Image prediction:
Acne

User symptoms:
red itchy patches on hands

This should normally be considered:

match_level = "MISMATCH"

and:

final_disease = "Inconclusive"

because the symptoms and body location conflict
with the image-model prediction.

5. Another example:

Image prediction:
Acne

User symptoms:
pimples and blackheads on face

This can be:

match_level = "MATCH"

and:

final_disease = "Acne"

provided the information is reasonably consistent.

6. If disease, symptoms, or body location clearly conflict:

match_level = "MISMATCH"

and:

final_disease MUST be:

"Inconclusive"

7. If information is insufficient:

match_level = "UNCERTAIN"

and:

final_disease should preferably be:

"Inconclusive"

8. If the image prediction and symptoms are reasonably
consistent:

match_level = "MATCH"

9. NEVER claim that the user definitely has a disease.

10. NEVER treat the image-model prediction as a confirmed
medical diagnosis.

11. If there is a mismatch, recommendations must focus on
general safe care and professional medical evaluation.

12. Do NOT give disease-specific treatment for the
image prediction when the image prediction conflicts
with the user's symptoms.

13. combined_confidence must represent confidence in the
COMBINED ASSESSMENT.

Do NOT simply copy the image confidence.

For example:

Image confidence = 99.59%

but image and symptoms clearly conflict.

The combined confidence should be LOW.

14. If match_level is MATCH:

combined_confidence can be a high value such as:

90%
95%
97%

depending on how strong the evidence is.

15. If match_level is MISMATCH:

combined_confidence should be LOW.

16. If match_level is UNCERTAIN:

combined_confidence should also be LOW or MODERATE.

17. Return ONLY valid JSON.

Return exactly this structure:

{{
    "final_disease": "",
    "match_level": "",
    "combined_confidence": "",
    "assessment": "",
    "care": "",
    "medicine": "",
    "doctor": ""
}}

Do not use Markdown.

Do not use ```json.

Do not add any text outside JSON.
"""

    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        result = response.text.strip()

        # ====================================================
        # REMOVE MARKDOWN IF GEMINI ADDS IT
        # ====================================================

        if result.startswith("```json"):
            result = result[7:]

        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        # ====================================================
        # JSON PARSE
        # ====================================================

        try:

            combined_result = json.loads(result)

            # =================================================
            # EXTRA SAFETY CHECK
            # =================================================

            if not isinstance(combined_result, dict):
                print("Combined AI result is not a dictionary.")
                return None

            required_keys = [
                "final_disease",
                "match_level",
                "combined_confidence",
                "assessment",
                "care",
                "medicine",
                "doctor"
            ]

            for key in required_keys:

                if key not in combined_result:
                    combined_result[key] = ""

            # =================================================
            # NORMALIZE MATCH LEVEL
            # =================================================

            match_level = str(
                combined_result.get(
                    "match_level",
                    ""
                )
            ).upper().strip()

            # =================================================
            # FINAL SAFETY RULE
            # =================================================

            if match_level in [
                "MISMATCH",
                "UNCERTAIN",
                "INCONCLUSIVE"
            ]:

                combined_result["final_disease"] = "Inconclusive"

            # =================================================
            # RETURN FINAL RESULT
            # =================================================

            return combined_result

        except json.JSONDecodeError:

            print(
                "Combined AI returned invalid JSON:"
            )

            print(result)

            return None

    except Exception as e:

        print(
            "Combined Analysis Error:",
            e
        )

        return None