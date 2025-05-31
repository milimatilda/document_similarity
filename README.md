# Text Similarity API with FastAPI & Sentence-BERT

This project is a FastAPI application that computes **semantic similarity** between two input texts using a pretrained Sentence-BERT model (`all-MiniLM-L6-v2`).

---

## Features

- Accepts two texts and returns a similarity score between 0 and 1.
- Uses `sentence-transformers` for sentence embeddings.
- Automatically splits long texts into manageable chunks (up to 500 characters), preserving sentence boundaries.
- Computes cosine similarity between average embeddings of the texts.

---

## Dependencies

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `scikit-learn`
- `pydantic`

## Running the API:

### Step 1: Clone the repository
```bash
git clone https://github.com/milimatilda/document_similarity.git
cd document_similarity
```
### Step 2: Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Run the FastAPI server
```bash
uvicorn app:app --reload
```

## Try it out in your browser:

Visit: http://127.0.0.1:8000/docs

You’ll find an interactive Swagger UI where you can test the API.


