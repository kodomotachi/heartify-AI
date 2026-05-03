
from operator import index
from xml.parsers.expat import model
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import logging
import time
import json
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "food-nutrition-recipes"
PINECONE_DIMENSION = 384  # all-MiniLM-L6-v2
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Dataset config
DATASET_NAME = "datahiveai/recipes-with-nutrition"
DATASET_SPLIT = "train"
MAX_SAMPLES = None  # Set to None for all data, or limit for testing (e.g., 1000)



ALLERGEN_KEYWORDS = {
    'dairy': ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'casein', 
              'cheddar', 'parmesan', 'gouda', 'brie', 'feta', 'mozzarella', 'ricotta'],
    'egg': ['egg', 'eggs', 'omelet', 'scrambled', 'mayonnaise'],
    'peanuts': ['peanut', 'peanuts'],
    'tree_nuts': ['almond', 'walnut', 'cashew', 'pistachio', 'hazelnut', 'pecan', 
                  'macadamia', 'pine nut', 'tahini'],
    'shellfish': ['shrimp', 'crab', 'lobster', 'prawn', 'crayfish', 'clam', 'mussel', 'oyster'],
    'fish': ['fish', 'salmon', 'tuna', 'cod', 'halibut', 'anchovy', 'sardine', 'trout'],
    'gluten': ['wheat', 'bread', 'flour', 'pasta', 'barley', 'rye', 'croissant', 
               'biscuit', 'pancake', 'muffin', 'burrito', 'tamale', 'roll'],
    'soy': ['soy', 'tofu', 'edamame', 'miso', 'tempeh', 'soy sauce']
}

DIET_KEYWORDS = {
    'vegan': ['vegan'],
    'vegetarian': ['vegetarian'],
    'gluten_free': ['gluten-free', 'gluten free'],
    'dairy_free': ['dairy-free', 'dairy free', 'lactose free'],
    'low_carb': ['low-carb', 'low carb', 'keto'],
    'high_protein': ['high-protein', 'high protein'],
    'paleo': ['paleo'],
    'mediterranean': ['mediterranean'],
    'whole30': ['whole30'],
    'pescatarian': ['pescatarian']
}


# ============================================
# HELPER FUNCTIONS TO EXTRACT NUTRITION DATA
# ============================================
def safe_join_list(x):
    """Safely join list to string"""
    if isinstance(x, list):
        return ', '.join([str(item) for item in x])
    if pd.isna(x):
        return ''
    return str(x)
    
def safe_parse_json(json_str):
    """Safely parse JSON string or return empty dict/list"""
    if pd.isna(json_str) or not json_str:
        return {}
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except:
        return {}


def extract_nutrient(total_nutrients, nutrient_code):
    """Extract nutrient value from total_nutrients dict"""
    if not total_nutrients or not isinstance(total_nutrients, dict):
        return 0.0
    
    nutrient = total_nutrients.get(nutrient_code, {})
    if isinstance(nutrient, dict):
        return float(nutrient.get('quantity', 0))
    return 0.0


def extract_ingredient_text(ingredient_lines):
    """Extract text from ingredient_lines (list of strings)"""
    if pd.isna(ingredient_lines):
        return ""
    
    if isinstance(ingredient_lines, list):
        return ', '.join([str(item) for item in ingredient_lines])
    elif isinstance(ingredient_lines, str):
        try:
            parsed = json.loads(ingredient_lines)
            if isinstance(parsed, list):
                return ', '.join([str(item) for item in parsed])
        except:
            pass
    
    return str(ingredient_lines)


def extract_ingredient_names(ingredients):
    if ingredients is None:
        return []

    # NaN scalar
    if isinstance(ingredients, float) and pd.isna(ingredients):
        return []

    # List case (dataset chuẩn)
    if isinstance(ingredients, list):
        return [
            str(item['food'])
            for item in ingredients
            if isinstance(item, dict) and 'food' in item
        ]

    # String JSON
    if isinstance(ingredients, str):
        try:
            parsed = json.loads(ingredients)
            if isinstance(parsed, list):
                return [
                    str(item.get('food', ''))
                    for item in parsed
                    if isinstance(item, dict)
                ]
        except Exception:
            pass

    return []



# ============================================
# LOAD DATASET FROM HUGGINGFACE
# ============================================
def load_recipe_dataset(dataset_name: str, split: str = "train", max_samples: int = None) -> pd.DataFrame:
    """Load dataset from HuggingFace"""
    logger.info(f" Loading dataset: {dataset_name} (split={split})")
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"   Limited to {max_samples} samples")
    
    df = pd.DataFrame(dataset)
    logger.info(f"Loaded {len(df)} recipes")
    logger.info(f"   Columns: {df.columns.tolist()}")
    
    # Show sample
    if 'recipe_name' in df.columns:
        logger.info(f"   Sample recipe: {df.iloc[0]['recipe_name']}")
    
    return df


# ============================================
# DETECT ALLERGENS FROM INGREDIENTS
# ============================================
def detect_allergens(ingredients_text: str) -> List[str]:
    """Detect allergens from ingredient text"""
    if not ingredients_text or pd.isna(ingredients_text):
        return []
    
    text_lower = str(ingredients_text).lower()
    detected = []
    
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected.append(allergen)
    
    return detected


# ============================================
# DETECT DIET TYPES
# ============================================
def detect_diet_types(recipe_name: str, ingredients: str, diet_labels: str, health_labels: str) -> List[str]:
    """Detect dietary classifications from recipe data"""
    combined_text = f"{recipe_name} {ingredients} {diet_labels} {health_labels}".lower()
    detected = []
    
    for diet, keywords in DIET_KEYWORDS.items():
        if any(keyword in combined_text for keyword in keywords):
            detected.append(diet)
    
    return detected


# ============================================
# CREATE RICH DESCRIPTION FOR EMBEDDING
# ============================================
def create_embedding_description(row: Dict) -> str:
    """
    Create rich text description for semantic search
    
    Format optimized for health/nutrition queries based on actual dataset structure:
    {recipe_name}. Nutrition: {calories} cal, {protein}g protein, {fat}g fat, 
    {carbs}g carbs, {fiber}g fiber. Ingredients: {ingredients}. 
    Serves {servings}. Diet: {diet_labels}. Health: {health_labels}.
    """
    parts = []
    
    # Recipe name 
    recipe_name = row.get('recipe_name', 'Unknown Recipe')
    parts.append(f"{recipe_name}")
    
    # Nutrition facts (extracted from total_nutrients)
    nutrition = []
    if row.get('calories') and row['calories'] > 0:
        nutrition.append(f"{row['calories']:.0f} calories")
    if row.get('protein_g') and row['protein_g'] > 0:
        nutrition.append(f"{row['protein_g']:.1f}g protein")
    if row.get('fat_g') and row['fat_g'] > 0:
        nutrition.append(f"{row['fat_g']:.1f}g fat")
    if row.get('carbohydrates_g') and row['carbohydrates_g'] > 0:
        nutrition.append(f"{row['carbohydrates_g']:.1f}g carbohydrates")
    if row.get('fiber_g') and row['fiber_g'] > 0:
        nutrition.append(f"{row['fiber_g']:.1f}g fiber")
    
    if nutrition:
        parts.append(f"Nutrition per serving: {', '.join(nutrition)}")
    
    # Ingredients (use ingredient names)
    ingredient_names = row.get('ingredient_names', [])
    if ingredient_names and isinstance(ingredient_names, list) and len(ingredient_names) > 0:
        ingredients_str = ', '.join(ingredient_names[:8])  # First 8 ingredients
        parts.append(f"Ingredients: {ingredients_str}")
    
    # Servings
    servings = row.get('servings', 0)
    if servings > 0:
        parts.append(f"Serves {int(servings)}")
    
    # Diet labels
    diet_labels = row.get('diet_labels', '')
    if diet_labels and len(str(diet_labels)) > 0:
        parts.append(f"Diet: {diet_labels}")
    
    # Health labels
    health_labels = row.get('health_labels', '')
    if health_labels and len(str(health_labels)) > 0:
        # Truncate if too long
        health_str = str(health_labels)[:200]
        parts.append(f"Health: {health_str}")
    
    # Meal type & Cuisine
    meal_type = row.get('meal_type', '')
    if meal_type:
        parts.append(f"Meal: {meal_type}")
    
    cuisine_type = row.get('cuisine_type', '')
    if cuisine_type:
        parts.append(f"Cuisine: {cuisine_type}")
    
    return '. '.join(parts)


# ============================================
# PROCESS DATASET
# ============================================
def process_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process recipes: extract nutrition, detect allergens, diets, create embeddings description
    """
    logger.info(f" Processing {len(df)} recipes...")
    
    # Parse JSON columns
    logger.info("   [1/7] Parsing JSON columns...")
    df['total_nutrients_parsed'] = df['total_nutrients'].apply(safe_parse_json)
    df['ingredients_parsed'] = df['ingredients'].apply(safe_parse_json)
    
    # Extract nutrition values from total_nutrients
    logger.info("   [2/7] Extracting nutrition values...")
    
    
    
    df['calories'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'ENERC_KCAL'))
    df['protein_g'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'PROCNT'))
    df['fat_g'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'FAT'))
    df['carbohydrates_g'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'CHOCDF'))
    df['fiber_g'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'FIBTG'))
    df['sugar_g'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'SUGAR'))
    df['sodium_mg'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'NA'))
    df['cholesterol_mg'] = df['total_nutrients_parsed'].apply(lambda x: extract_nutrient(x, 'CHOLE'))
    
    
    
    
    
    # Ensure servings is numeric
    df['servings'] = pd.to_numeric(df['servings'], errors='coerce').fillna(1)
    
    logger.info("   Normalizing nutrition per serving...")
    servings_safe = df['servings'].replace(0, 1)

    for col in [
    'calories',
    'protein_g',
    'fat_g',
    'carbohydrates_g',
    'fiber_g',
    'sugar_g',
    'sodium_mg',
    'cholesterol_mg'
  ]:
     df[col] = (df[col] / servings_safe).round(2)

    
    
    # Extract ingredient names FIRST (before using them)
    logger.info("   [3/7] Extracting ingredient names...")
    df['ingredient_names'] = df['ingredients_parsed'].apply(extract_ingredient_names)
    
    # Extract ingredient text for allergen detection
    logger.info("   [4/7] Extracting ingredient text...")
    df['ingredients_text'] = (
        df['ingredient_lines'].apply(extract_ingredient_text).fillna('') + ' ' +
        df['ingredient_names'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    ).str.strip()
    
    # Convert list columns to strings for display
    df['diet_labels'] = df['diet_labels'].apply(safe_join_list)
    df['health_labels'] = df['health_labels'].apply(safe_join_list)
    df['cautions'] = df['cautions'].apply(safe_join_list)
    df['cuisine_type'] = df['cuisine_type'].apply(safe_join_list)
    df['meal_type'] = df['meal_type'].apply(safe_join_list)
    df['dish_type'] = df['dish_type'].apply(safe_join_list)
    
    # Step 1: Detect allergens
    logger.info("   [5/7] Detecting allergens...")
    df['allergens'] = df['ingredients_text'].apply(detect_allergens)
    df['allergen_list'] = df['allergens'].apply(lambda x: ','.join(x) if x else 'none')
    
    # Create boolean flags for filtering
    for allergen in ALLERGEN_KEYWORDS.keys():
        df[f'has_{allergen}'] = df['allergens'].apply(lambda x: allergen in x)
    
    # Step 2: Detect diet types
    logger.info("   [6/7] Detecting diet types...")
    df['diet_types'] = df.apply(
        lambda row: detect_diet_types(
            row.get('recipe_name', ''),
            row.get('ingredients_text', ''),
            row.get('diet_labels', ''),
            row.get('health_labels', '')
        ),
        axis=1
    )
    
    # Create boolean flags
    for diet in DIET_KEYWORDS.keys():
        df[f'is_{diet}'] = df['diet_types'].apply(lambda x: diet in x)
    
    # Step 3: Calculate nutrition density score
    # Formula: (protein + fiber) / (calories + 1) * 100
    logger.info("   [7/7] Calculating nutrition scores & creating embeddings...")
    df['nutrition_density'] = (
        (df['protein_g'] + df['fiber_g']) / 
        (df['calories'] + 1) * 100
    ).round(2)
    
    # Step 4: Create embedding description
    df['description_for_embedding'] = df.apply(create_embedding_description, axis=1)
    
    # Stats
    logger.info(f"Processing complete:")
    logger.info(f"   - Recipes processed: {len(df)}")
    logger.info(f"   - Avg calories: {df['calories'].mean():.1f}")
    logger.info(f"   - Avg protein: {df['protein_g'].mean():.1f}g")
    logger.info(f"   - Avg nutrition density: {df['nutrition_density'].mean():.2f}")
    logger.info(f"   - Top allergens: {df['allergen_list'].value_counts().head(3).to_dict()}")
    
    return df


# ============================================
# INITIALIZE PINECONE
# ============================================
def initialize_pinecone(api_key: str, index_name: str, dimension: int, recreate: bool = False):
    """Initialize Pinecone and create/connect to index"""
    logger.info(f" Initializing Pinecone...")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing_indexes:
        if recreate:
            logger.info(f"   Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            time.sleep(5)
        else:
            logger.info(f"   Using existing index: {index_name}")
            return pc.Index(index_name)
    
    # Create new index
    logger.info(f"   Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        )
    )
    
    # Wait for index to be ready
    logger.info("   Waiting for index to be ready...")
    time.sleep(10)
    
    index = pc.Index(index_name)
    logger.info(f"Pinecone index ready: {index_name}")
    
    return index


# ============================================
# GENERATE EMBEDDINGS & UPLOAD
# ============================================
def upload_to_pinecone(
    df: pd.DataFrame,
    index,
    model: SentenceTransformer,
    batch_size: int = 100
) -> int:
    """
    Generate embeddings and upload to Pinecone
    """
    logger.info(f"Uploading {len(df)} recipes to Pinecone...")
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    uploaded_count = 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        
        # Generate embeddings for batch
        logger.info(f"   Batch {batch_num + 1}/{total_batches}: Generating embeddings...")
        embeddings = model.encode(
            batch['description_for_embedding'].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Prepare vectors
        vectors = []
        for idx, (_, row) in enumerate(batch.iterrows()):
            # Prepare metadata (Pinecone compatible)
            metadata = {
                # Basic info
                'recipe_name': str(row.get('recipe_name', 'Unknown'))[:500],
                'source': str(row.get('source', ''))[:200],
                'url': str(row.get('url', ''))[:500],
                
                # Nutrition (float)
                'calories': float(row.get('calories', 0)),
                'protein_g': float(row.get('protein_g', 0)),
                'fat_g': float(row.get('fat_g', 0)),
                'carbohydrates_g': float(row.get('carbohydrates_g', 0)),
                'fiber_g': float(row.get('fiber_g', 0)),
                'sugar_g': float(row.get('sugar_g', 0)),
                'sodium_mg': float(row.get('sodium_mg', 0)),
                'cholesterol_mg': float(row.get('cholesterol_mg', 0)),
                
                # Servings
                'servings': int(row.get('servings', 1)),
                'total_weight_g': float(row.get('total_weight_g', 0)),
                
                # Labels (strings - truncated)
                'diet_labels': str(row.get('diet_labels', ''))[:300],
                'health_labels': str(row.get('health_labels', ''))[:500],
                'cautions': str(row.get('cautions', ''))[:300],
                'cuisine_type': str(row.get('cuisine_type', ''))[:200],
                'meal_type': str(row.get('meal_type', ''))[:200],
                'dish_type': str(row.get('dish_type', ''))[:200],
                
                # Allergens
                'allergen_list': str(row.get('allergen_list', 'none')),
                'diet_types_list': ','.join(row.get('diet_types', [])),
                
                # Score
                'nutrition_density': float(row.get('nutrition_density', 0)),
                
                # Ingredients (truncated)
                'ingredients': str(row.get('ingredients_text', ''))[:1000],
            }
            
            # Add allergen boolean flags (for filtering)
            for allergen in ALLERGEN_KEYWORDS.keys():
                metadata[f'has_{allergen}'] = bool(row.get(f'has_{allergen}', False))
            
            # Add diet boolean flags
            for diet in DIET_KEYWORDS.keys():
                metadata[f'is_{diet}'] = bool(row.get(f'is_{diet}', False))
            
            # Generate ASCII-only ID (Pinecone requirement)
            # Remove non-ASCII characters from recipe name
            import re
            import unicodedata
            
            recipe_name = str(row.get('recipe_name', 'unknown'))
            # Normalize unicode characters (é -> e)
            normalized = unicodedata.normalize('NFKD', recipe_name)
            # Remove accents and keep only ASCII
            ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
            # Replace spaces and special chars with underscore
            clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', ascii_name)
            # Limit length and ensure uniqueness with index
            vector_id = f"recipe_{clean_name[:30]}_{row.name}"
            
            vectors.append({
                'id': vector_id,
                'values': embeddings[idx].tolist(),
                'metadata': metadata
            })
        
        # Upload batch to Pinecone
        try:
            index.upsert(vectors=vectors)
            uploaded_count += len(vectors)
            logger.info(
                f"   Batch {batch_num + 1}/{total_batches}: "
                f"Uploaded {len(vectors)} vectors ({uploaded_count}/{len(df)})"
            )
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.error(f"   ✗ Batch {batch_num + 1} failed: {e}")
            raise
    
    logger.info(f"Upload complete: {uploaded_count} recipes uploaded")
    return uploaded_count


# ============================================
# VERIFY UPLOAD
# ============================================
def verify_index(index):
    """Verify index stats"""
    logger.info("Verifying index...")
    
    stats = index.describe_index_stats()
    
    logger.info(f"Index Stats:")
    logger.info(f"   - Total vectors: {stats.get('total_vector_count', 0):,}")
    logger.info(f"   - Dimension: {stats.get('dimension', 0)}")
    logger.info(f"   - Index fullness: {stats.get('index_fullness', 0):.2%}")
    
    return stats
def check_vector_database(
    index,
    model: SentenceTransformer,
    query_text: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
):
   
    logger.info("=" * 60)
    logger.info("CHECKING VECTOR DATABASE")
    logger.info(f"Query: {query_text}")
    
    # 1. Embed query
    query_vector = model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()
    
    # 2. Query Pinecone
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    
    matches = response.get("matches", [])
    
    if not matches:
        logger.warning(" No results found!")
        return []
    
    # 3. Print results
    results = []
    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        
        result = {
            "rank": rank,
            "score": round(match.get("score", 0), 4),
            "recipe_name": metadata.get("recipe_name"),
            "calories": metadata.get("calories"),
            "protein_g": metadata.get("protein_g"),
            "diet_labels": metadata.get("diet_labels"),
            "allergens": metadata.get("allergen_list"),
            "cautions": metadata.get("cautions"),
            "nutrition_density": metadata.get("nutrition_density"),
            "url" : metadata.get("url")
        }
        
        results.append(result)
        
        logger.info(
            f"\n#{rank} | score={result['score']}\n"
            f"     {result['recipe_name']}\n"
            f"   Calories: {result['calories']}\n"
            f"   Protein: {result['protein_g']} g\n"
            f"   Diet: {result['diet_labels']}\n"
            f"   Allergens: {result['allergens']}\n"
            f"   Cautions: {result['cautions']}\n" 
            f"   Density: {result['nutrition_density']}"
            f"   URL: {result['url']}"
        )
    
    logger.info("=" * 60)
    return results


# ============================================
# MAIN PIPELINE
# ============================================
def main():
    """Run complete pipeline"""
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in environment")
    
    logger.info("=" * 70)
    logger.info("  PINECONE VECTOR DB PIPELINE - FOOD RECOMMENDATION")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Load dataset
    logger.info("\n[STEP 1] Loading dataset from HuggingFace...")
    df = load_recipe_dataset(DATASET_NAME, DATASET_SPLIT, MAX_SAMPLES)
    
    # Step 2: Process recipes
    logger.info("\n[STEP 2]   Processing recipes...")
    df = process_recipes(df)
    
    # Step 3: Initialize embedding model
    logger.info("\n[STEP 3] Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info(f"   Model loaded: all-MiniLM-L6-v2 ({PINECONE_DIMENSION}D)")
    
    # Step 4: Initialize Pinecone
    logger.info("\n[STEP 4]  Initializing Pinecone...")
    index = initialize_pinecone(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        recreate=True 
    )
    
    # Step 5: Upload to Pinecone
    logger.info("\n[STEP 5] Uploading to Pinecone...")
    uploaded = upload_to_pinecone(df, index, model, batch_size=100)
    
    # Step 6: Verify
    logger.info("\n[STEP 6] Verifying upload...")
    verify_index(index)
    
    logger.info("\n[STEP 7]  Checking vector database...")
    check_vector_database(
    index=index,
    model=model,
    query_text="high protein low carb chicken dinner",
    top_k=5
)
    # Complete
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETE!")
    logger.info(f"   - Total recipes: {len(df):,}")
    logger.info(f"   - Uploaded: {uploaded:,}")
    logger.info(f"   - Time: {elapsed:.1f}s ({uploaded/elapsed:.1f} recipes/sec)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()