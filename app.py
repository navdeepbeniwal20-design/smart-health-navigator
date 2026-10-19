
import os

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Gemini configuration
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

client = genai.Client(api_key=api_key) if api_key else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    # Accept the format used by your existing website
    message = data.get("question", data.get("message", ""))

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please enter your symptoms or question."
        }), 400

    message = message.strip()

    if len(message) > 2000:
        return jsonify({
            "error": "Please keep your question under 2,000 characters."
        }), 400

    if not api_key or client is None:
        return jsonify({
            "error": "Gemini API key is missing. Check your .env file."
        }), 503

    if not model_name:
        return jsonify({
            "error": "Gemini model ID is missing. Check your .env file."
        }), 503

    instructions = """
You are Smart Health Navigator, an educational health information
and symptom-navigation assistant for a student project.

Your goal is to help users understand symptoms, possible causes,
urgency, and appropriate next steps.

When a user reports symptoms:
1. Ask up to 3 brief, relevant follow-up questions if important
   details are missing, such as duration, severity, and other symptoms.
2. Explain possible causes without claiming a definite diagnosis.
3. Explain whether the user may need emergency, same-day,
   or routine medical attention.
4. Offer safe general self-care when appropriate.
5. Clearly explain warning signs that require urgent medical care.

For potentially life-threatening symptoms, including severe
breathing difficulty, chest pain, stroke warning signs, severe
bleeding, or loss of consciousness, advise immediate emergency
care rather than delaying with questions.

For emergencies in India, advise calling 112.
Do not prescribe medicines or doses.
Do not claim that serious conditions have been ruled out.
Use empathetic, simple language.
Do not request unnecessary personal information.
You are not a substitute for a healthcare professional.

For general health questions, answer them normally.
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=message,
            config={
                "system_instruction": instructions,
                "max_output_tokens": 500,
            },
        )

        reply = (response.text or "").strip()

        if not reply:
            return jsonify({
                "error": "No response received. Please try again."
            }), 502

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": (
                "Gemini request failed. Check your API key, model ID, "
                "internet connection, and terminal error."
            )
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
