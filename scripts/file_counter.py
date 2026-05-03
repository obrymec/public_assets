import os

# Define common image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
MATERIALS_EXTENSIONS = {".material", ".tres"}

def count_images(folder_path):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):  # Recursively walks through folders
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                image_count += 1
    return image_count

# Example usage
if __name__ == "__main__":
    folder = input("Enter folder path: ").strip()
    total_images = count_images(folder)
    print(f"Total number of images in '{folder}': {total_images}")

