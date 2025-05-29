from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()  # FastAPI application instance
model = SentenceTransformer('all-MiniLM-L6-v2')  #  Pretrained SentenceTransformer model for generating sentence embeddings


def convert_to_chunks(text, max_chunk_size = 500):
  """
  Splits the input text into chunks, each with a maximum length of `max_chunk_size`.
  
  Parameters:
    text (str): The input text to be chunked.
    max_chunk_size (int): The maximum allowed size of each chunk.

  Returns:
    List[str]: A list of text chunks.
  """
  # If the entire text is smaller than the maximum chunk size, return it as a single chunk.
  if len(text)<max_chunk_size:
    return [text.strip()]

  # Split the text into sentences based on the period delimiter.
  sentences = text.split('.')
  chunks = []  # List to store chunks of text
  current_chunk = ""
  
  for sentence in sentences:
    sentence = sentence.strip()

    # Ensures sentence ends with a period
    if not sentence.endswith('.'):
      sentence += '. ' 
    # Appends the sentence as a chunk, if sentence itself is long
    if len(sentence)>=max_chunk_size:
      chunks.append(sentence.strip())
    else:
      # Appends sentence/sentences ensuring the max_chunk_size is not exceeded
      if len(sentence) + len(current_chunk) >= max_chunk_size:
        chunks.append(current_chunk.strip())
        current_chunk = sentence 
      else:
        current_chunk += sentence
  # Appends the last chunk
  chunks.append(current_chunk.strip())
  return chunks



def get_mean_embedding(text_chunks):
  """
  Encodes the text_chunks to embeddings and returns the mean of the embeddings for the model
  """
  # Embedding for each chunk is encoded
  embeddings = model.encode(text_chunks, convert_to_tensor=True, normalize_embeddings=True)
  return embeddings.mean(dim=0)  # Returns the mean of the embeddings




# Define request format
class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/similarity")
def bert_similarity(input_texts: TextPair):

    # Both the input texts are converted to chunks of text
    chunk1 = convert_to_chunks(input_texts.text1)
    chunk2 = convert_to_chunks(input_texts.text2)
    
    # Converts the chunks to embeddings
    mean_embedding1 = get_mean_embedding(chunk1)
    mean_embedding2 = get_mean_embedding(chunk2)

    # Calculate the cosine similarity between the embeddings
    similarity = util.cos_sim(mean_embedding1, mean_embedding2).item()

    return {"similarity score" : similarity}

 