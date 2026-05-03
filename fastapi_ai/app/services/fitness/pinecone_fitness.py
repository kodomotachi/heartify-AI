"""
Fitness Exercise Recommendation System - Enhanced Pipeline
==============================================================

UPDATED VERSION with Intensity Classification + GIF URLs

This pipeline builds a vector database for semantic exercise search
with health-aware filtering capabilities AND intensity scoring.

NEW FEATURES:
- Automatic intensity classification (1-5 scale)
- Confidence scoring for classifications
- Safe intensity filtering based on user condition
- GIF URL storage for visual demonstrations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import json
from pathlib import Path
import os
from tqdm import tqdm
from intensity_classifier import IntensityClassifier, classify_dataset
from sentence_transformers import SentenceTransformer
import logging 
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CONFIGURATION

load_dotenv()

PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "index_name": "fitness-exercises",
    "dimension": 384,  # all-MiniLM-L6-v2
    "metric": "cosine",
    "cloud": "aws",
    "region": "us-east-1"
}


QUALITY_COLUMNS = {
    "core_fields": ["name", "bodyPart", "target", "equipment", "gifUrl"],  
    "secondary_muscles": ["secondaryMuscles/0", "secondaryMuscles/1"],
    "instructions": [f"instructions/{i}" for i in range(6)]
}


# =============================================================================
# DATA LOADING AND CLEANING (Updated)
# =============================================================================

def load_and_clean_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset and apply data quality constraints.
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} exercises")
    
    # Select quality columns
    all_columns = (
        QUALITY_COLUMNS["core_fields"] + 
        QUALITY_COLUMNS["secondary_muscles"] + 
        QUALITY_COLUMNS["instructions"]
    )
    
    if 'id' in df.columns:
        all_columns = ['id'] + all_columns
    
    df_clean = df[all_columns].copy()
    
    # Remove exercises missing critical fields 
    core_required = [col for col in QUALITY_COLUMNS["core_fields"] if col != "gifUrl"]
    df_clean = df_clean.dropna(subset=core_required)
    print(f"{len(df_clean)} exercises with complete core fields")
    
    # Fill missing gifUrl with empty string
    if 'gifUrl' in df_clean.columns:
        df_clean['gifUrl'] = df_clean['gifUrl'].fillna('')
    
    # ADD INTENSITY CLASSIFICATION
    print("\nClassifying exercise intensity...")
    df_clean = classify_dataset(df_clean)
    
    return df_clean


# TEXT EMBEDDING PREPARATION 

def merge_instructions(row: pd.Series) -> str:
    """Merge instruction steps."""
    instructions = []
    for i in range(6):
        col = f"instructions/{i}"
        if pd.notna(row[col]):
            step = str(row[col]).strip()
            instructions.append(f"Step {i+1}: {step}")
    return " ".join(instructions)


def merge_secondary_muscles(row: pd.Series) -> str:
    """Combine secondary muscles."""
    muscles = []
    for col in QUALITY_COLUMNS["secondary_muscles"]:
        if pd.notna(row[col]):
            muscles.append(str(row[col]).strip())
    return ", ".join(muscles) if muscles else "none"


def build_embedding_text(row: pd.Series) -> str:
    """
    Create rich semantic text for embedding.
    
    UPDATED: Now includes intensity information and visual demonstration note
    to improve semantic matching for energy-level queries.
    """
    instructions = merge_instructions(row)
    secondary = merge_secondary_muscles(row)
    intensity = row.get('intensity_label', 'MODERATE')
    has_visual = "Yes" if row.get('gifUrl', '') else "No"
    
    text = f"""Exercise: {row['name']}

Target Muscle: {row['target']}
Body Part: {row['bodyPart']}
Equipment Needed: {row['equipment']}
Intensity Level: {intensity}
Visual Demonstration Available: {has_visual}
Secondary Muscles: {secondary}

How to Perform:
{instructions}
"""
    return text.strip()


def prepare_embedding_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare exercises for embedding."""
    print("\nBuilding embedding texts...")
    
    df['embedding_text'] = df.apply(build_embedding_text, axis=1)
    df['secondary_muscles_merged'] = df.apply(merge_secondary_muscles, axis=1)
    
    # Count exercises with GIF URLs
    gif_count = df['gifUrl'].astype(bool).sum()
    print(f"Created {len(df)} embedding texts")
    print(f"Exercises with GIF URLs: {gif_count} ({gif_count/len(df)*100:.1f}%)")
    
    return df


# =============================================================================
# EMBEDDING GENERATION (Same as before)
# =============================================================================
def generate_embeddings(
    texts: List[str],
    batch_size: int = 64,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[List[float]]:
    """
    Generate embeddings using Sentence-Transformers (local model).
    
    Advantages:
    - No API key needed
    - Faster & cheaper
    - Deterministic results
    - Perfect for Pinecone retrieval
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} exercises...")
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True  
        )
        embeddings.extend(batch_embeddings.tolist())
    
    print(f"Generated {len(embeddings)} embeddings")
    return embeddings

# =============================================================================
# PINECONE INDEX SETUP (Same as before)
# =============================================================================

def initialize_pinecone() -> Pinecone:
    """Initialize Pinecone client."""
    pc = Pinecone(api_key=PINECONE_CONFIG["api_key"])
    return pc


def create_or_get_index(pc: Pinecone) -> Any:
    """Create or get Pinecone index."""
    index_name = PINECONE_CONFIG["index_name"]
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        print(f"\nCreating new index: {index_name}")
        
        pc.create_index(
            name=index_name,
            dimension=PINECONE_CONFIG["dimension"],
            metric=PINECONE_CONFIG["metric"],
            spec=ServerlessSpec(
                cloud=PINECONE_CONFIG["cloud"],
                region=PINECONE_CONFIG["region"]
            )
        )
        print(f"Index created successfully")
    else:
        print(f"\nIndex '{index_name}' already exists")
    
    return pc.Index(index_name)


# =============================================================================
# METADATA PREPARATION (Updated with GIF URL)
# =============================================================================

def build_metadata(row: pd.Series) -> Dict[str, Any]:
    """
    Build metadata for Pinecone filtering.
    
    UPDATED: Now includes gifUrl for visual demonstrations.
    """
    metadata = {
        # Core identification
        "exercise_name": row['name'],
        
        # Filtering fields
        "body_part": row['bodyPart'].lower(),
        "target_muscle": row['target'].lower(),
        "equipment": row['equipment'].lower(),
        "secondary_muscles": row['secondary_muscles_merged'].lower(),
        
        # Intensity fields
        "intensity_score": int(row['intensity_score']),
        "intensity_label": row['intensity_label'],
        "intensity_confidence": row['intensity_confidence'],
        
 
        "gif_url": row.get('gifUrl', ''),
        "has_visual": bool(row.get('gifUrl', '')),
        
        # Derived fields
        "has_equipment": row['equipment'].lower() != 'body weight',
        
        # Full text for display
        "full_instructions": row['embedding_text']
    }
    
    return metadata


# =============================================================================
# PINECONE UPSERT (Same as before)
# =============================================================================

def upsert_to_pinecone(
    index: Any, 
    df: pd.DataFrame, 
    embeddings: List[List[float]], 
    batch_size: int = 100
):
    """Upload vectors and metadata to Pinecone."""
    print(f"\nUpserting {len(df)} vectors to Pinecone...")
    
    vectors = []
    for idx, (_, row) in enumerate(df.iterrows()):
        vector_id = str(row['id']) if 'id' in row else f"ex_{idx}"
        
        vector = {
            "id": vector_id,
            "values": embeddings[idx],
            "metadata": build_metadata(row)
        }
        vectors.append(vector)
    
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"Upserted {len(vectors)} vectors")
    
    stats = index.describe_index_stats()
    print(f"Index stats: {stats['total_vector_count']} vectors")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(dataset_path: str):
    """
    Execute the complete Pinecone ingestion pipeline with intensity classification.
    """
    print("=" * 70)
    print("FITNESS EXERCISE RECOMMENDATION PIPELINE ")
    print("=" * 70)
    
    # Load and classify
    df = load_and_clean_dataset(dataset_path)
    
    # Show intensity distribution
    print("\nIntensity Distribution:")
    intensity_dist = df['intensity_label'].value_counts().sort_index()
    for label, count in intensity_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {label}: {count} exercises ({pct:.1f}%)")
    
    # Prepare texts
    df = prepare_embedding_data(df)
    
    # Generate embeddings
    embeddings = generate_embeddings(df['embedding_text'].tolist())
    
    # Initialize Pinecone
    pc = initialize_pinecone()
    index = create_or_get_index(pc)
    
    # Upsert data
    upsert_to_pinecone(index, df, embeddings)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"Index: {PINECONE_CONFIG['index_name']}")
    print(f"Vectors: {len(df)}")
    print("Features: Semantic search + Intensity filtering ")
    print("Ready for health-aware recommendations with visual guides!")
    
    return index


if __name__ == "__main__":
    DATASET_PATH = "../../data/megaGymDataset.csv"
    
    # Verify environment variables
    if not PINECONE_CONFIG["api_key"]:
        raise ValueError("PINECONE_API_KEY not set!")

    
    # Run the enhanced pipeline
    index = run_pipeline(DATASET_PATH)