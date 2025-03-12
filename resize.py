import os
import uuid
from PIL import Image
import argparse

def resize_images(input_path, output_path, size=(128, 128)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".png"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_path, f"{uuid.uuid4().hex}.png")
                
                try:
                    with Image.open(input_file) as img:
                        img = img.resize(size, Image.LANCZOS)
                        img.save(output_file, "PNG")
                    print(f"Processed: {input_file} -> {output_file}")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize PNG images and save to output directory with unique names.")
    parser.add_argument("input_path", help="Path to the input directory containing PNG files.")
    parser.add_argument("output_path", help="Path to the output directory where resized images will be saved.")
    
    args = parser.parse_args()
    resize_images(args.input_path, args.output_path)
