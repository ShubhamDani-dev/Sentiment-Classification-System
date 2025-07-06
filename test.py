#!/usr/bin/env python3
"""Simple API test script for sentiment analysis."""

import requests
from typing import List

def test_prediction(text: str) -> bool:
    """Test sentiment prediction for a given text."""
    try:
        response = requests.post("http://localhost:8000/predict", json={"text": text})
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        print(f"'{text}' -> {result['label']} ({result['score']:.3f})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False
    except (KeyError, ValueError) as e:
        print(f"Response parsing error: {e}")
        return False

def main() -> None:
    """Main test function."""
    print("Testing API...")
    
    test_cases: List[str] = [
        "This is amazing!",
        "I hate this product",
        "The weather is nice today",
        "This is terrible quality"
    ]
    
    for text in test_cases:
        test_prediction(text)

if __name__ == "__main__":
    main()
