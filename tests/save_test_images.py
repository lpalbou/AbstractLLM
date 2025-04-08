#!/usr/bin/env python
"""
Utility script to save test images for vision tests.

This script needs to be run directly from the command line to save the 
user-provided images to the tests/resources directory.
"""

import os
import sys
import requests
from io import BytesIO
import base64
from pathlib import Path
import argparse
from PIL import Image

# Define the resources directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define the test image files
TEST_IMAGES = [
    "mountain_path.jpg",     # Mountain path with wooden fence
    "city_park_sunset.jpg",  # City park with lampposts at sunset
    "humpback_whale.jpg",    # Humpback whale breaching
    "cat_carrier.jpg",       # Cat in pet carrier
]

def save_image_from_url(url, output_path):
    """
    Save an image from a URL to the specified path.
    
    Args:
        url: URL of the image
        output_path: Path where the image should be saved
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Successfully saved image to {output_path}")
            return True
        else:
            print(f"Failed to download image: HTTP status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error saving image from URL: {e}")
        return False

def save_image_from_file(source_path, output_path):
    """
    Save an image from a local file to the specified path.
    
    Args:
        source_path: Path of the source image
        output_path: Path where the image should be saved
    """
    try:
        # Copy the file
        import shutil
        shutil.copy2(source_path, output_path)
        print(f"Successfully copied image to {output_path}")
        return True
    except Exception as e:
        print(f"Error copying image: {e}")
        return False

def save_image_from_base64(base64_data, output_path):
    """
    Save an image from base64 data to the specified path.
    
    Args:
        base64_data: Base64 encoded image data
        output_path: Path where the image should be saved
    """
    try:
        # Remove data URI prefix if present
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]
            
        # Decode and save
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        print(f"Successfully saved image to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image from base64: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Save test images for vision tests")
    parser.add_argument("--source", required=True, help="Source of the image: 'url', 'file', or 'base64'")
    parser.add_argument("--input", required=True, help="URL, file path, or base64 data")
    parser.add_argument("--name", required=True, choices=TEST_IMAGES, help="Name to save the image as")
    
    args = parser.parse_args()
    
    output_path = os.path.join(RESOURCES_DIR, args.name)
    
    if args.source == "url":
        success = save_image_from_url(args.input, output_path)
    elif args.source == "file":
        success = save_image_from_file(args.input, output_path)
    elif args.source == "base64":
        success = save_image_from_base64(args.input, output_path)
    else:
        print(f"Unknown source type: {args.source}")
        sys.exit(1)
    
    if success:
        # Verify the image was saved correctly
        try:
            img = Image.open(output_path)
            print(f"Image verified: {img.format}, {img.size}, {img.mode}")
        except Exception as e:
            print(f"Warning: Could not verify saved image: {e}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 