from django.shortcuts import render
from django.conf import settings

import joblib
import os


# ==========================================================
# LOAD MODEL
# ==========================================================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "model.pkl"
)

VECTORIZER_PATH = os.path.join(
    settings.BASE_DIR,
    "vectorizer.pkl"
)

METRICS_PATH = os.path.join(
    settings.BASE_DIR,
    "metrics.pkl"
)


model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

metrics = joblib.load(
    METRICS_PATH
)


# ==========================================================
# HOME VIEW
# ==========================================================

def home(request):

    context = {

        # Model performance
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],

        "confusion_matrix": metrics[
            "confusion_matrix"
        ]

    }


    # ======================================================
    # EMAIL PREDICTION
    # ======================================================

    if request.method == "POST":

        message = request.POST.get(
            "message",
            ""
        ).strip()


        # --------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------

        if not message:

            context["error"] = (
                "Please enter an email message."
            )

            return render(
                request,
                "detector/index.html",
                context
            )


        # --------------------------------------------------
        # VECTORIZE MESSAGE
        # --------------------------------------------------

        message_vector = vectorizer.transform(
            [message]
        )


        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            message_vector
        )[0]


        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        probability = model.predict_proba(
            message_vector
        )[0]


        not_spam_probability = round(
            probability[0] * 100,
            2
        )

        spam_probability = round(
            probability[1] * 100,
            2
        )


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        if prediction == 1:

            result = "SPAM EMAIL"

            result_type = "spam"

        else:

            result = "NOT SPAM"

            result_type = "not-spam"


        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        confidence = round(
            max(probability) * 100,
            2
        )


        # --------------------------------------------------
        # UPDATE CONTEXT
        # --------------------------------------------------

        context.update({

            "message": message,

            "result": result,

            "result_type": result_type,

            "spam_probability":
                spam_probability,

            "not_spam_probability":
                not_spam_probability,

            "confidence":
                confidence,

        })


    return render(
        request,
        "detector/index.html",
        context
    )