import os
import random
import cv2
import numpy as np
from captcha.image import ImageCaptcha
from PIL import Image
from config import CHARS, CAPTCHA_LENGTH, WIDTH, HEIGHT, DATASET_SIZE

def apply_augmentations(image_path, label):
    """Applies blur and rotation to create augmented versions."""
    img = cv2.imread(image_path)
    
    # 1. Blur Version
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(image_path.replace(".png", "_blur.png"), blur_img)
    
    # 2. Rotated Version
    rows, cols, _ = img.shape
    angle = random.uniform(-10, 10) # Subtle rotation for captchas
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rot_img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(image_path.replace(".png", "_rot.png"), rot_img)

def generate():
    # Pathing logic to ensure it hits the root /dataset folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "dataset")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generator = ImageCaptcha(width=WIDTH, height=HEIGHT)
    
    print(f"🎨 Generating {DATASET_SIZE} base images + Augmentations...")
    
    for i in range(DATASET_SIZE):
        label = "".join(random.choices(CHARS, k=CAPTCHA_LENGTH))
        base_filename = f"{label}_{i}.png" # Added index to prevent collisions
        file_path = os.path.join(output_dir, base_filename)
        
        # Generate Base
        generator.write(label, file_path)
        
        # Apply Blur/Rotate (This brings us toward the TOTAL_SIZE goal)
        apply_augmentations(file_path, label)
        
        if i % 1000 == 0:
            print(f"Progress: {i}/{DATASET_SIZE} base images generated...")

    print(f"✅ Finished! Check your folder at: {output_dir}")

if __name__ == "__main__":
    # You may need: pip install opencv-python
    generate()
