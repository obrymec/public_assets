import os
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg", ".ico", ".heic", ".heif"}

def get_total_image_size(root_folder: str) -> None:
    # Whether the root folder exists and is a valid directory.
    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a valid directory.")
        sys.exit(1)

    total_bytes = 0
    image_count = 0

    # Walks through all subfolders recursively.
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:
                filepath = os.path.join(dirpath, filename)
                file_size = os.path.getsize(filepath)

                total_bytes += file_size
                image_count += 1

    # Converts bytes to a human-readable unit.
    units = [("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024), ("B", 1)]
    readable_size, unit_label = total_bytes, "B"

    for label, threshold in units:
        if total_bytes >= threshold:
            readable_size = total_bytes / threshold
            unit_label = label
            break

    print(f"Root folder : {os.path.abspath(root_folder)}")
    print(f"Images found: {image_count}")
    print(f"Total size  : {readable_size:.2f} {unit_label} ({total_bytes:,} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_size_calculator.py <root_folder>")
        sys.exit(1)

    get_total_image_size(sys.argv[1])
