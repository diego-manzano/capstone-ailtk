import os
import pickle
from typing import List
import google.generativeai as genai

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
        similarities = [self._jaccard_similarity(query, doc) for doc in self.corpus]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]
        return [self.corpus[i] for i in top_indices]

    def generate_augmented_response(self, user_prompt: str) -> str:
        # Generates a response using the LLM with an injected prompt from RAG results.
        similar_docs = self._get_similar_documents(user_prompt)
        injected_prompt = f"{user_prompt} {' '.join(similar_docs)}"
        response = self.model.generate_content(injected_prompt, stream=False)
        return response.text
