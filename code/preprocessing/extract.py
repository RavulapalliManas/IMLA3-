import os
from PIL import Image

# --- CONFIGURATION ---
# Get the project root dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(code_dir)

# 1. Where your current extracted files are
SOURCE_DIR = os.path.join(project_root, 'data', 'final_data')

# 2. Where you want the resized (224x224) files to go
DEST_DIR = os.path.join(project_root, 'data', 'final_data_224')

# 3. Target Dimension
IMG_SIZE = (224, 224)
# ---------------------

def preprocess_images():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print(f"Starting preprocessing...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {DEST_DIR}")
    print("-" * 30)

    success_count = 0
    error_count = 0

    # Walk through the extracted folder
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # Check if it's an image file (basic check)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                
                # Construct paths
                source_path = os.path.join(root, file)
                
                # Create corresponding subfolder in destination
                # relative_path gets us "Impressionism/monet.jpg"
                relative_path = os.path.relpath(source_path, SOURCE_DIR)
                dest_path = os.path.join(DEST_DIR, relative_path)
                
                # Ensure destination folder exists
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                try:
                    with Image.open(source_path) as img:
                        # 1. Convert to RGB 
                        # This fixes issues with PNG transparency (RGBA) or Grayscale (L)
                        img = img.convert('RGB')
                        
                        # 2. Resize
                        # Image.Resampling.LANCZOS is a high-quality downsampling filter
                        img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                        
                        # 3. Save
                        img_resized.save(dest_path, "JPEG", quality=90)
                        
                        success_count += 1
                        
                        if success_count % 100 == 0:
                            print(f"Processed {success_count} images...")

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    error_count += 1

    print("-" * 30)
    print("Preprocessing Complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Resized images are located in: {DEST_DIR}")

if __name__ == "__main__":
    preprocess_images()