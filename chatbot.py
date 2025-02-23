from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer  # type: ignore
import faiss  # type: ignore
import pandas as pd

app = Flask(__name__)

# Load model, index, and metadata at startup
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("schemes_index.faiss")
    metadata = pd.read_csv("scheme_metadata.csv")
except Exception as e:
    print(f"Error loading resources: {e}")
    raise

def retrieve_schemes(query, top_k=5):
    try:
        # Convert query to embedding
        query_embedding = model.encode([query])
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, top_k)
        
        # Retrieve corresponding schemes
        results = metadata.iloc[indices[0]]
        return results[["Scheme Name", "Objective", "Key Benefits", "Eligibility", "Source"]].to_dict(orient="records")
    except Exception as e:
        return {"error": f"Error retrieving schemes: {str(e)}"}

@app.route("/", methods=["GET", "POST"])
def home():
    schemes = []
    query = ""
    error = None
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = retrieve_schemes(query)
            if isinstance(results, dict) and "error" in results:
                error = results["error"]
            else:
                schemes = results
        else:
            error = "Please enter a valid query."

    return render_template("index.html", schemes=schemes, query=query, error=error)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)