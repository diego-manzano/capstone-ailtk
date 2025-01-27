import json
import os

import google.generativeai as genai
from flask import Flask, jsonify, request, send_file, send_from_directory

# 🔥🔥 FILL THIS OUT FIRST! 🔥🔥
# Get your Gemini API key by:
# - Selecting "Add Gemini API" in the "Project IDX" panel in the sidebar
# - Or by visiting https://g.co/ai/idxGetGeminiKey
API_KEY = 'AIzaSyCAMiRunlBvVWBpacIbiL4mc7ypN6xVnvo'

genai.configure(api_key=API_KEY)

app = Flask(__name__)


@app.route("/")
def index():
    return send_file('web/index.html')


@app.route("/api/generate", methods=["POST"])
def generate_api():
    if request.method == "POST":
        if API_KEY == 'TODO':
            return jsonify({ "error": '''
                To get started, get an API key at
                https://g.co/ai/idxGetGeminiKey and enter it in
                main.py
                '''.replace('\n', '') })
        try:
            req_body = request.get_json()
            content = req_body.get("contents")
            model = genai.GenerativeModel(model_name=req_body.get("model"))
            response = model.generate_content(content, stream=True)
            def stream():
                for chunk in response:
                    yield 'data: %s\n\n' % json.dumps({ "text": chunk.text })

            return stream(), {'Content-Type': 'text/event-stream'}

        except Exception as e:
            return jsonify({ "error": str(e) })


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)


if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 80)))


import mysql.connector
from mysql.connector import Error

# MySQL Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",  # Update with your DB host
        user="root",  # Update with your MySQL username
        password="",  # Update with your MySQL password
        database="ailtk_feedback"  # Update with your database name
    )

@app.route("/api/submit-feedback", methods=["POST"])
def submit_feedback():
    try:
        # Parse the incoming JSON data
        data = request.get_json()
        prompt = data.get("prompt", "")
        response = data.get("response", "")  # Model's response
        feedback_type = data.get("feedback_type", "")  # thumbs-up or thumbs-down
        additional_feedback = data.get("additional_feedback", "")  # Any additional feedback from the user

        # Log the feedback for debugging purposes
        print(f"Prompt: {prompt}")
        print(f"Model Response: {response}")  # Model's response 
        print(f"Feedback Type: {feedback_type}")  # thumbs-up or thumbs-down
        print(f"Additional Feedback: {additional_feedback}")  # Additional feedback from the user

        try:
            # Connect to the database
            conn = get_db_connection()
            cursor = conn.cursor()

            # Insert the feedback into the database
            query = """
            INSERT INTO entries (prompt, response, feedback_type, additional_feedback)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (prompt, response, feedback_type, additional_feedback))
            conn.commit()  # Save the changes

            return jsonify({"message": "Feedback saved to database successfully!"}), 200

        except Error as e:
            print(f"Error: {e}")
            return jsonify({"error": "Failed to save to database"}), 500

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    except Exception as e:
        print("Error processing feedback:", e)
        return jsonify({"error": str(e)}), 500