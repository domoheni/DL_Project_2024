#read the image folders and produse the csv s.
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the path to the dataset
base_path = "/Your/Path/Final_dataset/renamed_data"

# Initialize lists to store image paths and labels
image_paths = []
labels = []

# Load fake images
fake_dir = os.path.join(base_path, "fake")
fake_subfolders = os.listdir(fake_dir)

for subfolder in fake_subfolders:
    subfolder_path = os.path.join(fake_dir, subfolder)
    for img_file in os.listdir(subfolder_path):
        image_paths.append(os.path.join(subfolder_path, img_file))
        labels.append("fake")

# Load real images
real_dir = os.path.join(base_path, "real")
for subfolder in os.listdir(real_dir):
    subfolder_path = os.path.join(real_dir, subfolder)
    for img_file in os.listdir(subfolder_path):
        image_paths.append(os.path.join(subfolder_path, img_file))
        labels.append("real")

# Combine image paths and labels into a DataFrame
data = pd.DataFrame({"image": image_paths, "label": labels})

# Split data into training and testing datasets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Split fake images for each generator
for subfolder in fake_subfolders:
    fake_images = data[(data["label"] == "fake") & (data["image"].str.contains(subfolder))]
    fake_train, fake_test = train_test_split(fake_images, test_size=0.3, random_state=42)
    train_data = pd.concat([train_data, fake_train], ignore_index=True)
    test_data = pd.concat([test_data, fake_test], ignore_index=True)

# Split real images
real_images = data[data["label"] == "real"]
real_train, real_test = train_test_split(real_images, test_size=0.3, random_state=42)
train_data = pd.concat([train_data, real_train], ignore_index=True)
test_data = pd.concat([test_data, real_test], ignore_index=True)

# Save training and testing DataFrames to CSV files
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

# Create separate CSV files for fake test images by generator
for subfolder in fake_subfolders:
    fake_test_images = test_data[(test_data["label"] == "fake") & (test_data["image"].str.contains(subfolder))]
    output_csv = f"test_{subfolder}.csv"
    fake_test_images.to_csv(output_csv, index=False)

print("CSV files created successfully!")
