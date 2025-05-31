"""
Text Similarity API using Sentence-BERT and FastAPI

This FastAPI application exposes an API endpoint to compute the semantic similarity 
between two input texts using the SentenceTransformer model `all-MiniLM-L6-v2`.

Features:
- POST /similarity: Accepts two texts (`text1`, `text2`) and returns a similarity score.

Internally, the similarity logic is handled in `similarity.py`, which:
- Splits long texts into sentence-based chunks (max 500 characters).
- Computes mean sentence embeddings using a pretrained transformer model.
- Calculates cosine similarity between the averaged embeddings.

Example Request:
{
    "text1": "The quick brown fox jumps over the lazy dog.",
    "text2": "A fast, dark-colored fox leaped above a sleepy canine."
}

Response:
{
    "similarity score": 0.85
}

Dependencies:
- fastapi, pydantic, sentence-transformers
"""
from similarity import bert_similarity
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()  # FastAPI application instance


# Define request format
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/similarity")
def compute_similarity(input_texts: TextPair):
    """
    FastAPI route handler to compute semantic similarity between two input texts.

    This function:
    - Receives a JSON payload with two text strings ('text1' and 'text2').
    - Computes the similarity score using the 'bert_similarity' function.
    - Returns a JSON response with the similarity score.

    Args:
        input_texts (TextPair): A Pydantic model containing two strings: text1 and text2.

    Returns:
        dict: A JSON object with a single key "similarity score" and a float value.
    """
    similarity_score = bert_similarity(input_texts.text1, input_texts.text2)
    return {"similarity score": similarity_score}

 