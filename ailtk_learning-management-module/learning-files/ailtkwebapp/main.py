import json
import os
import google.generativeai as genai
from flask import Flask, jsonify, request, send_file, send_from_directory
from rag_orchestrator import RAGOrchestrator  # Import the RAGOrchestrator class
import pickle
from typing import List
import mysql.connector
from mysql.connector import Error


class RAGOrchestrator:
    # Manages corpus loading, similarity calculations, and generating augmented responses using the LLM.

    def __init__(self, pickle_file: str, model):
        # Initializes the RAGOrchestrator.
        self.pickle_file = pickle_file
        self.model = model
        self.corpus = self._load_corpus()

    def _load_corpus(self) -> List[str]:
        # Loads the corpus from a pickle file.
        if not os.path.exists(self.pickle_file):
            raise FileNotFoundError(f"Pickle file '{self.pickle_file}' not found. Please generate it first.")

        with open(self.pickle_file, "rb") as f:
            print("Corpus loaded from pickle file.")
            return pickle.load(f)

    @staticmethod
    def _jaccard_similarity(query: str, document: str) -> float:
        # Calculates Jaccard similarity between a query and a document.
        query_tokens = set(query.lower().split())
        document_tokens = set(document.lower().split())
        intersection = query_tokens.intersection(document_tokens)
        union = query_tokens.union(document_tokens)
        return len(intersection) / len(union)

    def _get_similar_documents(self, query: str, top_n: int = 5) -> List[str]:
        # Retrieves the top N most similar documents from the corpus.
        #  Ensure query is a string *here*:
        if isinstance(query, list):
            query = " ".join(map(str, query))  # Join list elements if needed
        else:
            query = str(query) #convert to string if it's another type
        similarities = [self._jaccard_similarity(query, doc) for doc in self.corpus] #query is now a string
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]
        return [self.corpus[i] for i in top_indices]

    def generate_augmented_response(self, user_prompt: str):
        # Generates a response using the LLM with an injected prompt from RAG results.
        similar_docs = self._get_similar_documents(user_prompt)
        injected_prompt = f"{user_prompt} {' '.join(similar_docs)}"
        response = self.model.generate_content(injected_prompt, stream=True)
        for chunk in response:
            yield chunk  # Yield the text chunks


# 🔥🔥 FILL THIS OUT FIRST! 🔥🔥
# Get your Gemini API key by:
# - Selecting "Add Gemini API" in the "Project IDX" panel in the sidebar
# - Or by visiting https://g.co/ai/idxGetGeminiKey
API_KEY = 'AIzaSyCAMiRunlBvVWBpacIbiL4mc7ypN6xVnvo'

genai.configure(api_key=API_KEY)

app = Flask(__name__)

# Define the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

MODEL = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="You are to serve as an AI virtual agent-coffee concierge for a company known as CoffeePro.\n    As a leading coffee retailer CoffeePro, aims to enhance their service of of selling wide\n    arrays coffee beans and blends from all around the world by providing personalized recommendations. \n\n    Given a user's preferences, such as:\n    * Drinking preference: Black or with milk/sugar\n    * Roast level: Light, medium, or dark\n    * Brew method: Espresso, pour over, cold brew, or French press\n    * Flavor profile: Fruity, nutty, chocolatey, or floral\n\n    You should:\n    1. Analyze the user's preferences and access your knowledge base of coffee beans to identify suitable options.\n    2. Provide detailed descriptions of recommended coffees, including their origin, flavor profile, and ideal brewing methods, based on the information provided from you in the injected prompts.\n    3. Offer personalized advice on brewing techniques, water temperature, and grind size to optimize the coffee experience.\n    4. Share interesting coffee facts and trivia to engage the user and foster a deeper appreciation for coffee.\n    5. Provide recommendations for food pairings that complement the coffee's flavor profile.\n    6. Answer questions about coffee history, roasting processes, and brewing techniques in a clear and informative manner.\n    7. Maintain a friendly and conversational tone to create a positive user experience. 8. Note that appended to the user prompt are info from a corpus using RAG, providing you information from company data to supplement your answer. DO NOT use info if not relevant to user prompt"
)

# Path to the pickle file
PICKLE_FILE = "corpus.pkl"

# Initialize RAGOrchestrator
orchestrator = RAGOrchestrator(PICKLE_FILE, MODEL)

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
            # model = genai.GenerativeModel(model_name=req_body.get("model"))
            response = orchestrator.generate_augmented_response(content)
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

# MySQL Database Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",  # Update with your DB host
        user="ailtk-learner",  # Update with your MySQL username
        password="DLSU1234!",  # Update with your MySQL password
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