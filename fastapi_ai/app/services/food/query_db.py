from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import logging

# ================= CONFIG =================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "food-nutrition-recipes"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= INIT =================
def load_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    return index

def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# ================= QUERY =================
def query_food(
    index,
    model,
    text: str,
    top_k: int = 5,
    filters: dict | None = None
):
    logger.info(f"Query: {text}")

    vector = model.encode(
        text,
        normalize_embeddings=True
    ).tolist()

    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )

    matches = res.get("matches", [])
    if not matches:
        print("No result")
        return

    for i, m in enumerate(matches, 1):
        md = m["metadata"]
        print(
            f"\n#{i} | score={m['score']:.4f}\n"
            f"{md.get('recipe_name')}\n"
            f"Calories: {md.get('calories')}\n"
            f"Protein: {md.get('protein_g')} g\n"
            f"Allergens: {md.get('allergen_list')}\n"
            f"Diet: {md.get('diet_labels')}"
        )

# ================= MAIN =================
if __name__ == "__main__":
    index = load_index()
    model = load_model()

    # TEST QUERIES
    query_food(
        index,
        model,
        text="healthy breakfast",
        filters={
        "has_gluten": False,
        "has_dairy": False,
        "is_vegan": True
    }
    )
