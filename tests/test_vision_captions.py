"""
Simple tests for evaluating vision model captions against keywords.

This script can be used to test image recognition quality without actually
running the models, using predefined captions.
"""

import os
import sys
import pytest
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define test image keywords
TEST_IMAGES = {
    "mountain_path": {
        "keywords": [
            "mountain", "mountains", "range", "dirt", "path", "trail", "wooden", "fence", 
            "sunlight", "sunny", "blue sky", "grass", "meadow", "hiking", 
            "countryside", "rural", "landscape", "horizon"
        ],
        "description": "A scenic mountain path with wooden fence on a sunny day"
    },
    "city_park_sunset": {
        "keywords": [
            "lamppost", "street light", "sunset", "dusk", "pink", "orange", "sky",
            "pathway", "walkway", "park", "urban", "trees", "buildings", "benches",
            "garden", "evening"
        ],
        "description": "A city park with lampposts at sunset with pink/orange sky"
    },
    "humpback_whale": {
        "keywords": [
            "whale", "humpback", "ocean", "sea", "breaching", "jumping", "splash",
            "marine", "mammal", "fins", "flipper", "gray", "waves", "wildlife",
            "water"
        ],
        "description": "A humpback whale breaching from the ocean"
    },
    "cat_carrier": {
        "keywords": [
            "cat", "pet", "carrier", "transport", "dome", "window", "plastic",
            "orange", "tabby", "fur", "eyes", "round", "opening", "white", "base",
            "ventilation", "air holes"
        ],
        "description": "A cat in a pet carrier with a dome window"
    }
}

# Sample captions for each image from different models
TEST_CAPTIONS = {
    "openai": {
        "mountain_path": """
        The image shows a beautiful mountain path winding through a high-altitude landscape. 
        Along the left side of the dirt trail is a rustic wooden fence. The path extends into the 
        distance where mountains can be seen on the horizon under a bright blue sky with scattered clouds. 
        The scene is bathed in sunlight, creating long shadows across the path. 
        There's some green vegetation visible along the sides of the path, and the overall 
        scenery suggests this is a hiking trail in a rural mountainous area, possibly in a 
        national park or nature preserve. The view captures the expansive landscape of 
        rolling mountain peaks extending into the distance.
        """,
        "city_park_sunset": """
        This image shows a beautiful park pathway at sunset or dusk. The path is lined with 
        vintage-style green lampposts that are illuminated. On both sides of the walkway are 
        trees and garden areas. The sky is a stunning display of pink and orange sunset colors, 
        creating a dramatic backdrop. In the distance, you can see urban buildings lining the 
        park. The scene has a peaceful evening atmosphere with the contrast between the urban 
        setting and natural park elements. The path appears to be in a city park, possibly in 
        Europe based on the architectural style visible in the background buildings and the 
        design of the lampposts.
        """,
        "humpback_whale": """
        The image shows a humpback whale breaching (jumping out of the water). The large marine 
        mammal is captured mid-breach with its massive body partially out of the ocean water. 
        You can see the distinctive ridged underside (ventral pleats) and one of its flippers. 
        The whale appears to be a dark gray color with some white markings typical of humpback whales. 
        Water is splashing around it as it emerges from the sea. The background shows the open ocean 
        with light blue sky visible above the horizon. This behavior is common among humpback whales 
        and is one of the most spectacular sights in wildlife viewing.
        """,
        "cat_carrier": """
        The image shows a cat inside a pet carrier or transport box. The cat, which appears to be 
        an orange/tabby color, is visible through a clear dome or bubble-like window on the top 
        of the carrier. The carrier itself is white with what look like ventilation holes at the 
        bottom. The cat's face and eyes are clearly visible looking out through the transparent 
        dome window. The background appears to have a patterned fabric or surface. This type of 
        pet carrier is designed for safely transporting cats or small animals while allowing them 
        to see outside.
        """
    },
    "anthropic": {
        "mountain_path": """
        This image shows a beautiful mountain landscape with a dirt path or trail winding through 
        it. On the left side of the path, there's a wooden fence running alongside it. The path 
        appears to be on elevated terrain, possibly a mountain or hill ridge, with magnificent 
        mountain ranges visible in the background extending to the horizon. The sky is bright blue 
        with some scattered white clouds, suggesting it's a clear, sunny day. The landscape has 
        patches of grass or meadow vegetation, and the overall scene has a rural, countryside feel 
        to it. This appears to be a hiking trail with a stunning panoramic view of the mountains.
        """,
        "city_park_sunset": """
        This image shows a beautiful park pathway or walkway at sunset. The path is lined with 
        vintage-style green lampposts that are lit up. On both sides of the walkway are trees 
        and landscaped areas. The sky is a stunning display of pink and orange sunset colors, 
        creating a dramatic backdrop. The scene has an urban park setting, with what appears to 
        be buildings visible in the background. The combination of the evening light, the lit 
        lampposts, and the garden setting creates a peaceful, romantic atmosphere. This appears 
        to be a public park in a city, captured during the magical "golden hour" around sunset.
        """,
        "humpback_whale": """
        This image shows a humpback whale breaching - jumping partially out of the water. The 
        massive marine mammal is captured mid-action as it emerges from the ocean surface. You 
        can see its distinctive dark gray body with the characteristic pleated underside. The 
        whale's pectoral fins or flippers are visible as it breaks through the water surface, 
        creating a splash around it. Humpback whales are known for this spectacular behavior 
        where they propel themselves out of the water. The background shows the open ocean water 
        stretching to the horizon. This is a wildlife photograph capturing one of nature's most 
        impressive sights - a large whale breaching in its natural marine habitat.
        """,
        "cat_carrier": """
        This image shows a cat inside what appears to be a pet carrier or transport container. 
        The cat, which looks to be an orange/ginger tabby, is visible through a clear plastic 
        dome or bubble window at the top of the carrier. The carrier itself is white with what 
        seem to be ventilation holes or air vents at the bottom. The cat's face is clearly visible 
        through the round transparent opening, and it appears to be looking out. This type of pet 
        carrier is designed for safely transporting cats or small animals while allowing them to 
        see their surroundings. The background shows some textured fabric with a pattern, possibly 
        the interior of a vehicle or home where the carrier is placed.
        """
    },
    "ollama": {
        "mountain_path": """
        The image shows a scenic mountain landscape with a dirt path or trail running through it. 
        On the left side of the path is a wooden fence. The path extends into the distance, with 
        mountains visible on the horizon under a blue sky with some clouds. The scene is bathed in 
        sunlight, creating shadows across the path. There appears to be some grassy or meadow areas 
        visible, and the overall setting suggests this is a hiking trail in a rural mountainous region. 
        The vista shows multiple mountain ranges extending into the distance, creating a beautiful 
        layered landscape effect.
        """,
        "city_park_sunset": """
        This image shows a park pathway or walkway at sunset or dusk. The path is lined with 
        several old-fashioned green lampposts that are illuminated. On either side of the path 
        are trees and planted areas. The sky has beautiful pink and orange sunset colors. In the 
        background, you can see what appear to be urban buildings. The scene has a peaceful evening 
        atmosphere in what looks like a city park. The combination of the warm sunset light, the 
        glowing lampposts, and the tree-lined path creates a picturesque urban garden setting.
        """,
        "humpback_whale": """
        The image shows a humpback whale breaching (jumping) out of the ocean water. The large 
        marine mammal is captured mid-breach with its body partially out of the water. You can 
        see its distinctive dark coloration and some of the characteristic features of a humpback 
        whale. There's water splashing around as the whale breaks the surface of the ocean. The 
        background shows the open sea extending to the horizon. Breaching is a common behavior 
        for humpback whales where they propel themselves partially or completely out of the water.
        """,
        "cat_carrier": """
        The image shows a cat inside what appears to be a pet carrier or transport container. 
        The cat, which seems to be orange or tabby colored, is visible through a clear dome or 
        bubble-like window at the top of the carrier. The carrier itself is white with what look 
        like ventilation holes at the bottom. The cat's face and eyes are clearly visible through 
        the round transparent opening. This type of pet carrier is designed for safely transporting 
        cats while allowing them to see out.
        """
    },
    "huggingface": {
        "mountain_path": """
        This image shows a rural mountain path or trail with a wooden fence running along the 
        left side. The dirt path winds forward into a landscape of mountains visible in the 
        distance under a blue sky with some clouds. The scene is sunny with good visibility 
        of the mountain ranges that extend to the horizon. There appears to be some grass or 
        meadow vegetation on the sides of the path. This appears to be a hiking trail in a 
        mountainous countryside area, possibly in a national park or nature preserve.
        """,
        "city_park_sunset": """
        The image shows a park pathway at sunset or dusk. The walkway is lined with green 
        lampposts that are lit up. On both sides of the path are trees and garden areas. 
        The sky has a beautiful pink and orange sunset glow. In the background, you can 
        see what appears to be city buildings. The scene captures an urban park in the 
        evening with the lampposts creating a pleasant ambiance along the path.
        """,
        "humpback_whale": """
        The image shows a humpback whale breaching (jumping) out of the ocean water. The 
        large marine mammal is captured mid-leap with its dark body partially emerged from 
        the water surface. You can see the distinctive features of the humpback whale, 
        including its flippers. Water is splashing around the whale as it breaks through 
        the surface. The background shows the open sea stretching to the horizon. This is 
        a typical breaching behavior where these large mammals propel themselves partially 
        out of the water.
        """,
        "cat_carrier": """
        The image shows a cat inside what appears to be a pet carrier or transport container. 
        The cat, which looks to be orange or tabby colored, is visible through a clear dome 
        or bubble-like window on the top of the carrier. The carrier itself is white with 
        what seem to be ventilation holes visible at the bottom. The cat's face is clearly 
        visible through the round transparent opening. This type of pet carrier is designed 
        for safely transporting cats while allowing them to see out.
        """
    }
}


def calculate_keyword_match_percentage(caption: str, keywords: List[str]) -> float:
    """
    Calculate the percentage of keywords found in the caption.
    
    Args:
        caption: The caption text to check
        keywords: List of keywords to look for
        
    Returns:
        Percentage of keywords found (0.0 to 1.0)
    """
    if not caption or not keywords:
        return 0.0
        
    caption_lower = caption.lower()
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in caption_lower]
    return len(matched_keywords) / len(keywords)


def evaluate_caption(caption: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Evaluate a caption against keywords.
    
    Args:
        caption: The caption text to evaluate
        keywords: List of keywords to look for
        
    Returns:
        Dictionary with evaluation results
    """
    match_percentage = calculate_keyword_match_percentage(caption, keywords)
    
    # Determine the quality level
    if match_percentage >= 0.75:
        quality = "excellent"
    elif match_percentage >= 0.5:
        quality = "good"
    elif match_percentage >= 0.25:
        quality = "fair"
    else:
        quality = "poor"
        
    return {
        "match_percentage": match_percentage,
        "quality": quality,
        "matched_keywords": [k for k in keywords if k.lower() in caption.lower()],
        "missed_keywords": [k for k in keywords if k.lower() not in caption.lower()]
    }


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "huggingface"])
@pytest.mark.parametrize("image_name", ["mountain_path", "city_park_sunset", "humpback_whale", "cat_carrier"])
def test_caption_quality(provider, image_name):
    """Test the quality of captions against image keywords."""
    # Get the caption and keywords
    caption = TEST_CAPTIONS[provider][image_name]
    keywords = TEST_IMAGES[image_name]["keywords"]
    description = TEST_IMAGES[image_name]["description"]
    
    # Evaluate the caption
    evaluation = evaluate_caption(caption, keywords)
    
    # Log the results
    logger.info(f"\nTesting {provider} caption for {image_name} ({description}):")
    logger.info(f"Caption quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
    logger.info(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
    logger.info(f"Matched: {', '.join(evaluation['matched_keywords'][:5])}{'...' if len(evaluation['matched_keywords']) > 5 else ''}")
    logger.info(f"Missed: {', '.join(evaluation['missed_keywords'][:5])}{'...' if len(evaluation['missed_keywords']) > 5 else ''}")
    
    # Assert minimum quality level
    assert evaluation["match_percentage"] >= 0.25, f"Caption quality too low: {evaluation['quality']}"
    

def test_keyword_comparison():
    """
    Compare all models' performance for each image, showing which models
    performed best at identifying the key elements.
    """
    results = {}
    
    # Evaluate all captions
    for image_name in TEST_IMAGES.keys():
        results[image_name] = {}
        keywords = TEST_IMAGES[image_name]["keywords"]
        
        for provider in TEST_CAPTIONS.keys():
            caption = TEST_CAPTIONS[provider][image_name]
            evaluation = evaluate_caption(caption, keywords)
            results[image_name][provider] = evaluation
    
    # Print comparison table
    logger.info("\n===== Model Comparison (% of keywords matched) =====")
    
    # Header
    header = f"{'Image':<20} | {'OpenAI':<10} | {'Anthropic':<10} | {'Ollama':<10} | {'HuggingFace':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Data rows
    for image_name in TEST_IMAGES.keys():
        row = f"{image_name:<20} | "
        for provider in ["openai", "anthropic", "ollama", "huggingface"]:
            percentage = results[image_name][provider]["match_percentage"] * 100
            row += f"{percentage:>8.1f}% | "
        logger.info(row)
    
    # Summary - best model per image
    logger.info("\n===== Best Performance By Image =====")
    for image_name in TEST_IMAGES.keys():
        # Sort providers by match percentage
        sorted_providers = sorted(
            ["openai", "anthropic", "ollama", "huggingface"],
            key=lambda p: results[image_name][p]["match_percentage"],
            reverse=True
        )
        
        best_provider = sorted_providers[0]
        best_score = results[image_name][best_provider]["match_percentage"] * 100
        
        logger.info(f"{image_name}: {best_provider} ({best_score:.1f}%)")
    
    # Overall winner
    totals = {}
    for provider in ["openai", "anthropic", "ollama", "huggingface"]:
        totals[provider] = sum(results[img][provider]["match_percentage"] for img in TEST_IMAGES.keys())
    
    overall_best = max(totals.items(), key=lambda x: x[1])
    logger.info(f"\nOverall best model: {overall_best[0]} (avg: {overall_best[1]/len(TEST_IMAGES)*100:.1f}%)")
    
    # Assert there's a clear winner
    assert overall_best[1] > 0, "No clear winner in model comparison"


if __name__ == "__main__":
    # Enable more detailed logging for direct script execution
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_keyword_comparison() 