#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
import os

# Create a test image with the right size for LLaVA models (336x336)
img_size = (336, 336)
img = Image.new("RGB", img_size, (240, 240, 240))
draw = ImageDraw.Draw(img)

# Draw some shapes
draw.rectangle([(20, 20), (img_size[0]-20, img_size[1]-20)], outline=(200, 0, 0), width=5)
draw.ellipse([(50, 50), (img_size[0]-50, img_size[1]-50)], outline=(0, 0, 200), width=5)
draw.line([(20, 20), (img_size[0]-20, img_size[1]-20)], fill=(0, 150, 0), width=5)
draw.line([(20, img_size[1]-20), (img_size[0]-20, 20)], fill=(0, 150, 0), width=5)

# Add text
text = f"Test Image {img_size[0]}x{img_size[1]}"
draw.text((img_size[0]//2 - 80, img_size[1]//2 - 10), text, fill=(0, 0, 0))

# Save the image
os.makedirs("tests/examples", exist_ok=True)
filename = f"tests/examples/test_image_{img_size[0]}x{img_size[1]}.jpg"
img.save(filename)
print(f"Created test image: {filename}") 