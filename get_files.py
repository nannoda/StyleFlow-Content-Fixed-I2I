import os
import sys

def find_png_files(directory, output_file="png_files.txt"):
    """Recursively finds all .png files in the given directory and writes their absolute paths to a text file."""
    with open(output_file, "w") as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".png"):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    f.write(abs_path + "\n")
    print(f"Paths saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print("Error: Provided path is not a directory.")
        sys.exit(1)
    
    find_png_files(directory_path)