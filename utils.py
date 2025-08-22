import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")

def find_similar_articles(articles, embeddings, target_index, similarity_threshold=0.87):
    if target_index < 0 or target_index >= len(articles):
        raise ValueError(f"Target index {target_index} is out of range (0-{len(articles)-1})")
    
    target_embedding = embeddings[target_index].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    similar_indices = np.where(similarities >= similarity_threshold)[0]
    similar_indices = [idx for idx in similar_indices if idx != target_index]
    similar_articles = [
        {
            'article': {**{k: v for k, v in articles[idx].items() if k != 'text'},
            'text': articles[idx]['text'][:100]},
            'similarity': float(similarities[idx]),
            'index': int(idx)
        }
        for idx in similar_indices
    ]
    similar_articles.sort(key=lambda x: x['similarity'], reverse=True)

    return similar_articles


def load_articles_from_file(file_path):
    articles = []
    skipped = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    text = item.get('text', '')

                    if text:
                        articles.append({
                            "index": len(articles),
                            "text": text
                        })
                    else:
                        skipped += 1
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_num} in {file_path}")
                    skipped += 1

    print(f"Loaded {len(articles)} valid articles from {file_path}")
    print(f"Skipped {skipped} articles with empty text or invalid JSON")

    return articles


def generate_embeddings(data, model):
    print(f"Generating embeddings for {len(data)} articles sequentially...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    embeddings = []
    metadata = []

    for i, item in enumerate(tqdm(data)):
        content = item.get('content', '')
        with torch.no_grad():
            embedding = model.encode(
                content,
                convert_to_numpy=True,
            )

        embeddings.append(embedding)
        metadata.append({
          "index": i,  
          "title": item.get("title", ""),
          "site": item.get("site", "")
      })
    embeddings_array = np.array(embeddings)

    return embeddings_array, metadata

def generate_embed(text):
  try:
      if isinstance(text, str):
          embedding = model.encode(text)
      else:
          # Handle non-string case
          embedding = np.zeros(model.get_sentence_embedding_dimension())
  except Exception as e:
      print(f"Error computing embedding: {e}")
      embedding = np.zeros(model.get_sentence_embedding_dimension())
  return embedding


def process_text(text, prompt):
        # Remove prompt from beginning if present
        if isinstance(text, str) and isinstance(prompt, str) and text.startswith(prompt):
            processed = text[len(prompt):].strip()
        else:
            processed = text

        return processed