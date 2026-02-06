from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import logging

# ================= CONFIG =================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "fitness-exercises"   # index bạn vừa test
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= INIT =================
def load_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# ================= QUERY =================
def query_exercises(
    index,
    model,
    text: str,
    top_k: int = 5,
    filters: dict | None = None,
    max_intensity: int | None = None
):
    """
    Query fitness exercises from Pinecone.
    """
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

    print("\n" + "=" * 80)
    print(f"QUERY: {text}")
    print("=" * 80)

    shown = 0
    for i, m in enumerate(matches, 1):
        md = m["metadata"]
        intensity = md.get("intensity_score", 3)

        # optional agent safety filter
        if max_intensity is not None and intensity > max_intensity:
            continue

        shown += 1
        print(
            f"\n#{shown} | score={m['score']:.4f}\n"
            f"Exercise: {md.get('exercise_name')}\n"
            f"Target: {md.get('target_muscle')}\n"
            f"Body part: {md.get('body_part')}\n"
            f"Equipment: {md.get('equipment')}\n"
            f"Intensity: {md.get('intensity_label')} ({intensity})\n"
        )

    if shown == 0:
        print("All results filtered out by intensity")

# ================= MAIN =================
if __name__ == "__main__":
    index = load_index()
    model = load_model()

    # ------------------------------------------------
    # TEST 1 – BASIC SEMANTIC SEARCH
    # ------------------------------------------------
    query_exercises(
        index,
        model,
        text="upper body exercises that do not require leg movement",
        filters={
            "equipment": {"$eq": "body weight"},
            "body_part": {"$nin": ["upper legs", "lower legs"]}
        },
        
    )

    # # ------------------------------------------------
    # # TEST 2 – BODYWEIGHT ONLY
    # # ------------------------------------------------
    # query_exercises(
    #     index,
    #     model,
    #     text="upper body workout at home",
    #     filters={
    #         "equipment": {"$eq": "body weight"}
    #     }
    # )

    # # ------------------------------------------------
    # # TEST 3 – LOW INTENSITY (TIRED USER)
    # # ------------------------------------------------
    # query_exercises(
    #     index,
    #     model,
    #     text="gentle exercises for tired body",
    #     max_intensity=2
    # )

    # # ------------------------------------------------
    # # TEST 4 – MULTI CONSTRAINT
    # # ------------------------------------------------
    # query_exercises(
    #     index,
    #     model,
    #     text="my leg are hurt so give me an upper body workout",
    #     # filters={
    #     #     "equipment": {"$eq": "body weight"},
    #     #     "body_part": {"$nin": ["upper legs", "lower legs"]}
    #     # },
    #     max_intensity=2
    # )
