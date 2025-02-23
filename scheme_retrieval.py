from sentence_transformers import SentenceTransformer   # type: ignore
import faiss  # type: ignore
import pandas as pd

# Load model, index, and metadata
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("schemes_index.faiss")
metadata = pd.read_csv("scheme_metadata.csv")

def retrieve_schemes(query, top_k=5):  # Changed top_k to 5
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve corresponding schemes
    results = metadata.iloc[indices[0]]
    return results[["Scheme Name", "Objective", "Key Benefits", "Eligibility", "Source"]]

# Example usage
query = "What schemes are available for small farmers in Maharashtra?"
results = retrieve_schemes(query)
print(results)