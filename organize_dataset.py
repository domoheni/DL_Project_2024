import os
import shutil
import numpy as np

# Paths to your dataset
real_data_dir = "./Stanford_Cars/stanford-cars-real-train-fewshot"
synthetic_data_dir = "./Stanford_Cars/stanford-cars-synthetic-classwise-16/synthetic_16"
output_dir = "./data"

# Prepare directories for train/val/test split
splits = ['train', 'val', 'test']
categories = ['0_real', '1_fake']  # Updated for DIRE format
count = 0

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, "stanford_cars", category), exist_ok=True)

# Function to split and organize dataset
def organize_data(src_dir, dest_dir, split, category, val_ratio=0.1, test_ratio=0.1):
    class_names = sorted(os.listdir(src_dir))
    for class_name in class_names:
        class_path = os.path.join(src_dir, class_name)
        if os.path.isdir(class_path):
            filenames = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))]
            total = len(filenames)
            val_count = int(val_ratio * total)
            test_count = int(test_ratio * total)
            train_count = total - val_count - test_count
            
            # Shuffle filenames for random splits
            np.random.shuffle(filenames)
            
            # Split into train, val, test
            for idx, filename in enumerate(filenames):
                file_path = os.path.join(class_path, filename)
                if idx < train_count:
                    split_dir = os.path.join(dest_dir, 'train', "stanford_cars", category)
                elif idx < train_count + val_count:
                    split_dir = os.path.join(dest_dir, 'val', "stanford_cars", category)
                else:
                    split_dir = os.path.join(dest_dir, 'test', "stanford_cars", category)
                
                # Copy image to the correct split directory
                shutil.copy(file_path, os.path.join(split_dir, f"{class_name}_{filename}"))

# Organize real and synthetic images
organize_data(real_data_dir, output_dir, 'train', '0_real')
organize_data(synthetic_data_dir, output_dir, 'train', '1_fake')
organize_data(real_data_dir, output_dir, 'val', '0_real')
organize_data(synthetic_data_dir, output_dir, 'val', '1_fake')
organize_data(real_data_dir, output_dir, 'test', '0_real')
organize_data(synthetic_data_dir, output_dir, 'test', '1_fake')
