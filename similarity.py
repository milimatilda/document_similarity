from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  #  Pretrained SentenceTransformer model for generating sentence embeddings


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



def bert_similarity(text1, text2):
    """
    Compute semantic similarity score between two input texts.

    Steps:
    - Splits each text into chunks.
    - Encodes chunks to embeddings and averages them.
    - Calculates cosine similarity between the averaged embeddings.

    Args:
        text1, text2: Input texts 'text1' and 'text2' strings.

    Returns:
        similarity (float): The similarity score as a float.
    """

    # Both the input texts are converted to chunks of text
    chunk1 = convert_to_chunks(text1)
    chunk2 = convert_to_chunks(text2)
    
    # Converts the chunks to embeddings
    mean_embedding1 = get_mean_embedding(chunk1)
    mean_embedding2 = get_mean_embedding(chunk2)

    # Calculate the cosine similarity between the embeddings
    similarity = util.cos_sim(mean_embedding1, mean_embedding2).item()

    return round(similarity, 2)
