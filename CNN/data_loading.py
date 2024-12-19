import os
import shutil

# Define the paths
source_dir = "/dtu/blackhole/10/203248/dataset/test_data"
destination_dir = "/dtu/blackhole/06/203238/dl_project/cnn/test_data/renamed_data"

# Create the destination directory
os.makedirs(destination_dir, exist_ok=True)

# Traverse through the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):  # Include image file types
            # Get the relative path and parent folder name
            relative_path = os.path.relpath(root, source_dir)
            parent_folder = os.path.basename(root)

            # Construct the new file name
            new_file_name = f"{parent_folder}_{file}"

            # Define the source and destination file paths
            src_path = os.path.join(root, file)
            dest_folder = os.path.join(destination_dir, relative_path)
            os.makedirs(dest_folder, exist_ok=True)  # Create the destination folder
            dest_path = os.path.join(dest_folder, new_file_name)

            # Copy and rename the file
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {src_path} -> {dest_path}")

print("Dataset restructuring and renaming completed.")