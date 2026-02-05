"""
Exercise Intensity Classifier


Automatically classify exercise intensity/difficulty based on:
- Exercise name keywords
- Equipment used
- Body parts involved
- Movement patterns in instructions

Intensity levels:
- VERY_LOW (1): Stretching, gentle mobility
- LOW (2): Basic bodyweight, light cardio
- MODERATE (3): Standard strength training
- HIGH (4): Compound movements, weighted exercises
- VERY_HIGH (5): Plyometrics, explosive, olympic lifts
"""

import re
from typing import Dict, List, Tuple
import pandas as pd


# =============================================================================
# INTENSITY CLASSIFICATION RULES
# =============================================================================

class IntensityClassifier:
    """
    Rule-based intensity classification for exercises.
    """
    
    # Keywords that indicate intensity levels
    INTENSITY_KEYWORDS = {
        "VERY_LOW": [
            "stretch", "stretching", "mobility", "neck", "wrist", "ankle",
            "rotation", "circle", "roll", "gentle", "breathing"
        ],
        "LOW": [
            "walk", "march", "toe", "heel", "arm circle", "leg swing",
            "standing", "seated", "lying", "raise", "curl", "extension"
        ],
        "MODERATE": [
            "push", "pull", "press", "row", "squat", "lunge", "bridge",
            "plank", "crunch", "sit-up", "dip", "chin"
        ],
        "HIGH": [
            "deadlift", "bench press", "barbell", "heavy", "weighted",
            "pull-up", "muscle-up", "pistol", "jump squat", "box jump"
        ],
        "VERY_HIGH": [
            "plyometric", "explosive", "sprint", "burpee", "thruster",
            "snatch", "clean", "jerk", "box jump", "depth jump",
            "jump rope double", "mountain climber sprint"
        ]
    }
    
    # Equipment-based intensity hints
    EQUIPMENT_INTENSITY = {
        "body weight": 2, 
        "dumbbell": 3,
        "barbell": 4,
        "cable": 3,
        "kettlebell": 3,
        "band": 2,
        "medicine ball": 3,
        "resistance band": 2,
        "smith machine": 3,
        "leverage machine": 3,
        "rope": 2,
        "stability ball": 2,
        "olympic barbell": 5,
        "trap bar": 4,
        "ez barbell": 3
    }
    
    # Body part complexity (multiple large muscle groups = higher intensity)
    HIGH_INTENSITY_BODYPARTS = [
        "upper legs",  # Squats, lunges - very taxing
        "cardio"       # High heart rate
    ]
    
    # Movement patterns in instructions that indicate intensity
    MOVEMENT_PATTERNS = {
        "VERY_LOW": ["hold", "slowly", "gently", "relax"],
        "LOW": ["lift", "lower", "raise", "extend"],
        "MODERATE": ["push", "pull", "squeeze", "contract"],
        "HIGH": ["explode", "power", "maximum", "heavy"],
        "VERY_HIGH": ["jump", "sprint", "burst", "explosive", "fast as possible"]
    }
    
    def __init__(self):
        self.intensity_map = {
            "VERY_LOW": 1,
            "LOW": 2,
            "MODERATE": 3,
            "HIGH": 4,
            "VERY_HIGH": 5
        }
    
    def _score_by_keywords(self, text: str) -> int:
        """
        Score intensity based on keywords in exercise name.
        Returns intensity level (1-5).
        """
        text_lower = text.lower()
        scores = []
        
        for intensity, keywords in self.INTENSITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores.append(self.intensity_map[intensity])
        
        # Return max score if found, otherwise moderate
        return max(scores) if scores else 3
    
    def _score_by_equipment(self, equipment: str) -> int:
        """
        Score intensity based on equipment used.
        """
        equipment_lower = equipment.lower()
        
        # Check for exact matches first
        if equipment_lower in self.EQUIPMENT_INTENSITY:
            return self.EQUIPMENT_INTENSITY[equipment_lower]
        
        # Check for partial matches
        for equip, score in self.EQUIPMENT_INTENSITY.items():
            if equip in equipment_lower:
                return score
        
        # Default to moderate
        return 3
    
    def _score_by_bodypart(self, bodypart: str) -> int:
        """
        Score intensity based on body part complexity.
        """
        bodypart_lower = bodypart.lower()
        
        # Large muscle groups or cardio = higher intensity potential
        if any(hp in bodypart_lower for hp in self.HIGH_INTENSITY_BODYPARTS):
            return 4
        
        # Isolation movements = moderate
        if bodypart_lower in ["neck", "waist", "upper arms", "lower arms"]:
            return 2
        
        return 3
    
    def _score_by_instructions(self, instructions: str) -> int:
        """
        Score intensity based on movement patterns in instructions.
        """
        if not instructions or pd.isna(instructions):
            return 3
        
        instructions_lower = instructions.lower()
        scores = []
        
        for intensity, patterns in self.MOVEMENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in instructions_lower:
                    scores.append(self.intensity_map[intensity])
        
        return max(scores) if scores else 3
    
    def _detect_special_cases(self, name: str, instructions: str) -> Tuple[bool, int]:
        """
        Detect special high/low intensity cases.
        Returns (is_special, intensity_level)
        """
        text = (name + " " + str(instructions)).lower()
        
        # Very high intensity patterns
        very_high_patterns = [
            "olympic", "plyometric", "explosive", "sprint", "burpee",
            "max effort", "as fast as possible", "amrap"
        ]
        
        for pattern in very_high_patterns:
            if pattern in text:
                return True, 5
        
        # Very low intensity patterns
        very_low_patterns = [
            "stretch", "mobility", "breathing", "meditation",
            "gentle", "relaxation"
        ]
        
        for pattern in very_low_patterns:
            if pattern in text:
                return True, 1
        
        return False, 3
    
    def classify(
        self,
        name: str,
        equipment: str,
        bodypart: str,
        instructions: str = ""
    ) -> Dict[str, any]:
        """
        Classify exercise intensity.
        
        Returns:
            {
                'intensity_score': 1-5,
                'intensity_label': 'VERY_LOW' | 'LOW' | 'MODERATE' | 'HIGH' | 'VERY_HIGH',
                'confidence': 'high' | 'medium' | 'low',
                'reasoning': List of factors
            }
        """
        # Check special cases first
        is_special, special_score = self._detect_special_cases(name, instructions)
        
        if is_special:
            return {
                'intensity_score': special_score,
                'intensity_label': self._score_to_label(special_score),
                'confidence': 'high',
                'reasoning': ['Special case detected']
            }
        
        # Score from multiple factors
        scores = {
            'name': self._score_by_keywords(name),
            'equipment': self._score_by_equipment(equipment),
            'bodypart': self._score_by_bodypart(bodypart),
            'instructions': self._score_by_instructions(instructions)
        }
        
        # Weighted average (name and instructions are most important)
        weights = {
            'name': 2.0,
            'equipment': 1.5,
            'bodypart': 1.0,
            'instructions': 1.5
        }
        
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        total_weight = sum(weights.values())
        final_score = round(weighted_sum / total_weight)
        
        # Ensure score is in range
        final_score = max(1, min(5, final_score))
        
        # Determine confidence based on score agreement
        score_values = list(scores.values())
        score_range = max(score_values) - min(score_values)
        
        if score_range <= 1:
            confidence = 'high'
        elif score_range <= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Build reasoning
        reasoning = []
        for factor, score in scores.items():
            if score != 3:  # Only mention non-neutral factors
                reasoning.append(f"{factor}: {self._score_to_label(score)}")
        
        return {
            'intensity_score': final_score,
            'intensity_label': self._score_to_label(final_score),
            'confidence': confidence,
            'reasoning': reasoning if reasoning else ['Based on general patterns']
        }
    
    def _score_to_label(self, score: int) -> str:
        """Convert numeric score to label."""
        labels = {1: 'VERY_LOW', 2: 'LOW', 3: 'MODERATE', 4: 'HIGH', 5: 'VERY_HIGH'}
        return labels.get(score, 'MODERATE')
    
    def get_safe_intensity_for_condition(self, condition: str) -> int:
        """
        Get maximum safe intensity for a health condition.
        
        Args:
            condition: 'tired', 'heart_condition', 'beginner', 'injury', etc.
        
        Returns:
            Maximum safe intensity score (1-5)
        """
        safe_levels = {
            'exhausted': 1,
            'tired': 2,
            'low_energy': 2,
            'heart_condition': 2,
            'cardiac_issue': 2,
            'beginner': 3,
            'novice': 3,
            'injury': 2,
            'recovering': 2,
            'elderly': 2,
            'pregnancy': 2,
            'chronic_pain': 2
        }
        
        return safe_levels.get(condition.lower(), 3)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def classify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add intensity classification to entire dataset.
    
    Args:
        df: DataFrame with columns: name, equipment, bodyPart, instructions
    
    Returns:
        DataFrame with added columns:
        - intensity_score
        - intensity_label
        - intensity_confidence
    """
    classifier = IntensityClassifier()
    
    results = []
    
    print(f"Classifying intensity for {len(df)} exercises...")
    
    for _, row in df.iterrows():
        # Merge instructions if multiple columns
        instructions = ""
        for i in range(10):
            col = f"instructions/{i}"
            if col in row and pd.notna(row[col]):
                instructions += " " + str(row[col])
        
        classification = classifier.classify(
            name=row['name'],
            equipment=row['equipment'],
            bodypart=row['bodyPart'],
            instructions=instructions.strip()
        )
        
        results.append(classification)
    
    # Add results to dataframe
    df['intensity_score'] = [r['intensity_score'] for r in results]
    df['intensity_label'] = [r['intensity_label'] for r in results]
    df['intensity_confidence'] = [r['confidence'] for r in results]
    
    print("Intensity classification complete!")
    
    # Print distribution
    print("\nIntensity Distribution:")
    print(df['intensity_label'].value_counts().sort_index())
    
    return df


# =============================================================================
# HELPER FUNCTIONS FOR AGENT USE
# =============================================================================

def filter_by_max_intensity(
    exercises: List[Dict],
    max_intensity: int
) -> List[Dict]:
    """
    Filter exercises by maximum intensity level.
    
    Args:
        exercises: List of exercise dicts with 'metadata' key
        max_intensity: Maximum allowed intensity (1-5)
    
    Returns:
        Filtered list of exercises
    """
    filtered = []
    
    for ex in exercises:
        metadata = ex.get('metadata', {})
        intensity = metadata.get('intensity_score', 3)
        
        if intensity <= max_intensity:
            filtered.append(ex)
    
    return filtered


def get_intensity_recommendation(user_state: str) -> Dict[str, any]:
    """
    Get recommended intensity range for user state.
    
    Args:
        user_state: 'tired', 'energetic', 'normal', 'recovering', etc.
    
    Returns:
        {
            'max_intensity': int,
            'recommended_range': (min, max),
            'avoid_labels': List[str]
        }
    """
    classifier = IntensityClassifier()
    
    recommendations = {
        'exhausted': {
            'max_intensity': 1,
            'recommended_range': (1, 1),
            'avoid_labels': ['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']
        },
        'tired': {
            'max_intensity': 2,
            'recommended_range': (1, 2),
            'avoid_labels': ['HIGH', 'VERY_HIGH']
        },
        'low_energy': {
            'max_intensity': 2,
            'recommended_range': (1, 2),
            'avoid_labels': ['HIGH', 'VERY_HIGH']
        },
        'normal': {
            'max_intensity': 4,
            'recommended_range': (2, 4),
            'avoid_labels': ['VERY_HIGH']
        },
        'energetic': {
            'max_intensity': 5,
            'recommended_range': (3, 5),
            'avoid_labels': []
        },
        'beginner': {
            'max_intensity': 3,
            'recommended_range': (1, 3),
            'avoid_labels': ['HIGH', 'VERY_HIGH']
        },
        'intermediate': {
            'max_intensity': 4,
            'recommended_range': (2, 4),
            'avoid_labels': ['VERY_HIGH']
        },
        'advanced': {
            'max_intensity': 5,
            'recommended_range': (2, 5),
            'avoid_labels': []
        }
    }
    
    return recommendations.get(user_state.lower(), recommendations['normal'])


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Single exercise classification
    print("="*80)
    print("EXAMPLE 1: Single Exercise Classification")
    print("="*80)
    
    classifier = IntensityClassifier()
    
    exercises = [
        {
            'name': 'burpee',
            'equipment': 'body weight',
            'bodypart': 'cardio',
            'instructions': 'Jump down, push up, jump up explosively'
        },
        {
            'name': 'push-up',
            'equipment': 'body weight',
            'bodypart': 'chest',
            'instructions': 'Lower body slowly, push back up'
        },
        {
            'name': 'neck stretch',
            'equipment': 'body weight',
            'bodypart': 'neck',
            'instructions': 'Gently tilt head to side, hold stretch'
        },
        {
            'name': 'barbell squat',
            'equipment': 'barbell',
            'bodypart': 'upper legs',
            'instructions': 'Lower with heavy weight, explode up'
        }
    ]
    
    for ex in exercises:
        result = classifier.classify(
            ex['name'], ex['equipment'], ex['bodypart'], ex['instructions']
        )
        print(f"\n{ex['name'].upper()}")
        print(f"  Intensity: {result['intensity_label']} ({result['intensity_score']}/5)")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Reasoning: {', '.join(result['reasoning'])}")
    
    # Example 2: User state recommendations
    print("\n" + "="*80)
    print("EXAMPLE 2: User State Recommendations")
    print("="*80)
    
    user_states = ['tired', 'normal', 'energetic', 'beginner']
    
    for state in user_states:
        rec = get_intensity_recommendation(state)
        print(f"\n{state.upper()} user:")
        print(f"  Max intensity: {rec['max_intensity']}/5")
        print(f"  Recommended range: {rec['recommended_range']}")
        print(f"  Avoid: {', '.join(rec['avoid_labels']) if rec['avoid_labels'] else 'None'}")
    
    # Example 3: Dataset classification (if dataset available)
    print("\n" + "="*80)
    print("EXAMPLE 3: Dataset Classification")
    print("="*80)
    
    try:
        df = pd.read_csv("megaGymDataset.csv")
        df_classified = classify_dataset(df.head(100))  # Test on first 100
        
        print("\nSample classifications:")
        sample = df_classified[['name', 'equipment', 'intensity_label', 'intensity_score']].head(10)
        print(sample.to_string(index=False))
        
    except FileNotFoundError:
        print("Dataset not found - skipping example 3")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)