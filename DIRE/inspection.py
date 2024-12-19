import numpy as np
import os
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to the directory where the .npy files are stored
results_dir = "/Users/heni/Documents/DTU/3_Semester/DeepLearning/Model"
dataset = "/Users/heni/Documents/DTU/3_Semester/DeepLearning/test"

# Load the .npy files
y_true = np.load(os.path.join(results_dir, f"test_y_true.npy"))
y_pred = np.load(os.path.join(results_dir, f"test_y_pred.npy"))

# Handle filenames
try:
    filenames = np.load(os.path.join(results_dir, f"test_filenames.npy"))
    if len(filenames) == 0:
        raise ValueError("Filenames are empty. Extracting dynamically.")
except Exception as e:
    print(f"Error with filenames: {e}")
    # Dynamically generate file paths and folder names
    filenames = [os.path.join(root, file)
                 for root, _, files in os.walk(dataset)
                 for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Extracted {len(filenames)} filenames dynamically.")

# Extract folder names
folder_names = [os.path.basename(os.path.dirname(filepath)) for filepath in filenames]

# Inspect the contents
print("First 10 y_true:", y_true[:10])
print("First 10 y_pred:", y_pred[:10])
print("First 10 filenames:", filenames[:10])
print("First 10 folder names:", folder_names[:10])

# Convert predictions to binary
y_pred_binary = y_pred > 0.5

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_binary)
average_precision = average_precision_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.5f}")
print(f"Average Precision: {average_precision:.5f}")

# Overall Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)
ConfusionMatrixDisplay(conf_matrix, display_labels=["Real (0)", "Fake (1)"]).plot(cmap="Blues")
plt.title("Overall Confusion Matrix")
plt.show()

# Analyze predictions by folder
results_by_folder = defaultdict(list)
for folder, true_label, pred_label in zip(folder_names, y_true, y_pred_binary):
    results_by_folder[folder].append((true_label, pred_label))

# Folder-Wise Analysis and Confusion Matrices
# Folder-Wise Confusion Matrices
for folder, results in results_by_folder.items():
    folder_y_true = [r[0] for r in results]
    folder_y_pred = [r[1] for r in results]
    
    print(f"\nFolder: {folder}")
    print(f"  Accuracy: {accuracy_score(folder_y_true, folder_y_pred):.5f}")
    
    # Folder-specific confusion matrix with correct labels
    folder_conf_matrix = confusion_matrix(folder_y_true, folder_y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(folder_conf_matrix, display_labels=["Real (0)", "Fake (1)"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix for {folder}")
    plt.show()

# Overall Prediction Distribution
plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Real Class (0)')
plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Fake Class (1)')
plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Prediction Distribution by Class')
plt.legend()
plt.show()
