# Imports --------------------------------------------------------------------------------------------------------------
import os
import random
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, ToPILImage
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from torchviz import make_dot
from PIL import Image
import hiddenlayer as hl


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#LOADING IMAGES --------------------------------------------------------------------------------------------------------------

# Set the seed for reproducibility
random.seed(42)

# Define paths
base_path = "/dtu/blackhole/06/203238/dl_project/cnn/test_data/renamed_data"
fake_path = os.path.join(base_path, "fake")
real_path = os.path.join(base_path, "real")

# Initialize lists for train, validation, and test datasets
train_data = []
val_data = []
test_data = []

# Helper function to split data
def split_data(files, label, train_data, val_data, test_data):
    random.shuffle(files)
    n_total = len(files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    train_data.extend([(file, label) for file in train_files])
    val_data.extend([(file, label) for file in val_files])
    test_data.extend([(file, label) for file in test_files])

# Process fake images
for subdir in os.listdir(fake_path):
    subdir_path = os.path.join(fake_path, subdir)
    if os.path.isdir(subdir_path):
        files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        split_data(files, "fake", train_data, val_data, test_data)

# Process real images
for subdir in os.listdir(real_path):
    subdir_path = os.path.join(real_path, subdir)
    if os.path.isdir(subdir_path):
        files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        split_data(files, "real", train_data, val_data, test_data)

# Convert to DataFrames
train_df = pd.DataFrame(train_data, columns=["file_path", "label"])
val_df = pd.DataFrame(val_data, columns=["file_path", "label"])
test_df = pd.DataFrame(test_data, columns=["file_path", "label"])

# Save to CSV files
output_dir = "/dtu/blackhole/06/203238/dl_project/cnn/Final_csvs"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled DataFrame back to a CSV file
val_df.to_csv('validation_shuffled_output.csv', index=False)


train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled DataFrame back to a CSV file
train_df.to_csv('training_shuffled_output.csv', index=False)


test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled DataFrame back to a CSV file
test_df.to_csv('testing_shuffled_output.csv', index=False)

print("Data successfully split and saved!")


def display_examples(data, num_examples=5):
    """
    Display a few examples from the dataset.

    Args:
        data (list): A list of tuples (file_path, label).
        num_examples (int): Number of examples to display.
    """
    plt.figure(figsize=(15, 5))
    for i in range(num_examples):
        file_path, label = data[i]
        # Load image using OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
        
        # Display the image
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.savefig("images_display.png")  # Save plot to a file
    plt.close()

# Display examples from the training dataset
display_examples(train_data, num_examples=5)

# Display examples from the validation dataset
display_examples(val_data, num_examples=5)


# MODEL CREATION --------------------------------------------------------------------------------------------------------------

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        #Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) 

        #sigmoid for BCELoss
        x = torch.sigmoid(x)
        return x

deepModel = DeepCNN().to(device)
criterionDeep = nn.BCELoss()  # Binary Cross-Entropy Loss (expects probabilities)
optimizerDeep = optim.AdamW(deepModel.parameters(), lr=0.001, weight_decay=1e-4)
# DATASET CREATION --------------------------------------------------------------------------------------------------------------

print("Creating Dataset Class")

class RealFakeDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): List of tuples (file_path, label).
            transform (callable, optional): Transform to be applied to the images.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        
        
        image = Image.open(file_path).convert('RGB') 
        
        
        if self.transform:
            image = self.transform(image)
        
       
        label = 0 if label == "real" else 1 
        label = torch.tensor(label, dtype=torch.float32) 

        return image, label
    


transform = transforms.Compose([    
    transforms.Resize((128, 128)),       
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),              
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize datasets
train_dataset = RealFakeDataset(train_df.values.tolist(), transform=transform)
val_dataset = RealFakeDataset(val_df.values.tolist(), transform=val_test_transform)
test_dataset = RealFakeDataset(test_df.values.tolist(), transform=val_test_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



test_augment = Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

augmented_test_dataset = RealFakeDataset(test_df.values.tolist(), transform=test_augment)
augmented_test_loader = DataLoader(augmented_test_dataset, batch_size=32, shuffle=False)



# TRAINING AND VALIDATION --------------------------------------------------------------------------------------------------------------

mean = torch.tensor([0.5, 0.5, 0.5]) 
std = torch.tensor([0.5, 0.5, 0.5])   

def denormalize(image):
    image = image * std[:, None, None] + mean[:, None, None]
    return image.clamp(0, 1)

# Training and Validation
print("--------------Testing and Validation-----------------------------------------------------------------------------------------")
num_epochs = 10
train_losses = []
val_losses = []
scheduler = StepLR(optimizerDeep, step_size=5, gamma=0.1)

best_val_loss = float('inf')
patience = 3 
epochs_no_improve = 0

for epoch in range(num_epochs):
    # Training phase
    deepModel.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

        optimizerDeep.zero_grad()
        outputs = deepModel(inputs)
        loss = criterionDeep(outputs, labels)
        loss.backward()
        optimizerDeep.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation phase
    deepModel.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = deepModel(inputs)
            preds = (outputs > 0.5).float()
            loss = criterionDeep(outputs, labels)
            running_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_losses.append(running_loss / len(val_loader))
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Early stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        epochs_no_improve = 0
        torch.save(deepModel.state_dict(), "best_model.pth")  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    # Update scheduler
    scheduler.step()


# Plot Training and Validation Loss
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.savefig("Training_Validation_Loss.png") 
plt.close()

# Load the best model for evaluation
deepModel.load_state_dict(torch.load("best_model.pth"))
deepModel.eval()

# Evaluation on Validation Set
y_true_val = []
y_pred_val = []
y_probs_val = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = deepModel(inputs)
        preds = (outputs > 0.5).float()
        y_true_val.extend(labels.cpu().numpy())
        y_pred_val.extend(preds.cpu().numpy())
        y_probs_val.extend(outputs.cpu().numpy())

# Validation Metrics
print("Validation Accuracy:", accuracy_score(y_true_val, y_pred_val))
print("Validation Classification Report:\n", classification_report(y_true_val, y_pred_val))
cm = confusion_matrix(y_true_val, y_pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("Validation_Confusion_Matrix.png")
plt.close()

# TESTING --------------------------------------------------------------------------------------------------------------
# Evaluation on Test Set
print("--------------Testing (Test Set)-----------------------------------------------------------------------------------------")
y_true_test = []
y_pred_test = []
y_probs_test = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = deepModel(inputs)
        preds = (outputs > 0.5).float()
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds.cpu().numpy())
        y_probs_test.extend(outputs.cpu().numpy())

# Test Metrics
print("Test Accuracy:", accuracy_score(y_true_test, y_pred_test))
print("Test Classification Report:\n", classification_report(y_true_test, y_pred_test))
cm = confusion_matrix(y_true_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("Test_Confusion_Matrix.png")  # Save plot to a file
plt.close()

# AUC-ROC
print("Test AUC-ROC:", roc_auc_score(y_true_test, np.array(y_probs_test).flatten()))

# Visualizing Predictions
for i in range(10):
    image = val_dataset[i][0]  # Get normalized image
    label = val_dataset[i][1]
    prediction = deepModel(image.unsqueeze(0).to(device)).item()

    # Denormalize for visualization
    image = denormalize(image)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(f"True: {'Real' if label == 0 else 'Fake'}, Predicted: {'Real' if prediction < 0.5 else 'Fake'}")
    plt.savefig(f"Prediction{i}.png")  # Save plot to a file
    plt.close()


print("--------------Testing (Augmented Test Set)-----------------------------------------------------------------------------------------")
y_true_test = []
y_pred_test = []
y_probs_test = []

with torch.no_grad():
    for inputs, labels in augmented_test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = deepModel(inputs)
        preds = (outputs > 0.5).float()
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds.cpu().numpy())
        y_probs_test.extend(outputs.cpu().numpy())

# Test Metrics
print("Test Accuracy:", accuracy_score(y_true_test, y_pred_test))
print("Test Classification Report:\n", classification_report(y_true_test, y_pred_test))
cm = confusion_matrix(y_true_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("Test_Confusion_Matrix.png")  # Save plot to a file
plt.close()

# AUC-ROC
print("Test AUC-ROC:", roc_auc_score(y_true_test, np.array(y_probs_test).flatten()))

# Visualizing Predictions
for i in range(10):
    image = val_dataset[i][0]  # Get normalized image
    label = val_dataset[i][1]
    prediction = deepModel(image.unsqueeze(0).to(device)).item()

    # Denormalize for visualization
    image = denormalize(image)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(f"True: {'Real' if label == 0 else 'Fake'}, Predicted: {'Real' if prediction < 0.5 else 'Fake'}")
    plt.savefig(f"Prediction{i}.png")  # Save plot to a file
    plt.close()


print("-------------- Architecture -----------------------------------------------------------------------------------------")

x = torch.randn(1, 3, 128, 128).to(device)
y = deepModel(x) 
dot = make_dot(y, params=dict(deepModel.named_parameters()))
dot.render("model_architecture", format="png") 


'''
INCOMPATIBLE PYTORCH VERSION :((((((((((((

# Pass the model and example input
dummy_input = torch.randn(1, 3, 128, 128).to(device)
hl_graph = hl.build_graph(deepModel, dummy_input)
hl_graph.save("model_graph", format="png")

'''


print("-------------- Confusion Matrixes -----------------------------------------------------------------------------------------")

def extract_folder_name(file_path):
    """Extracts the folder name from the file path."""
    return os.path.basename(os.path.dirname(file_path))

def plot_confusion_matrices_by_folder(y_true, y_pred, file_paths, labels):
    """
    Plots confusion matrices for each folder based on the folder names extracted from file paths.
    
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        file_paths (list): List of file paths for the images.
        labels (list): List of class labels (e.g., ["Real", "Fake"]).
    """
    assert len(y_true) == len(y_pred) == len(file_paths), \
        "y_true, y_pred, and file_paths must have the same length"

    # Extract folder names
    folder_names = [extract_folder_name(path) for path in file_paths]

    # Group by folder
    unique_folders = sorted(set(folder_names))
    folder_to_indices = {folder: [] for folder in unique_folders}
    for idx, folder in enumerate(folder_names):
        folder_to_indices[folder].append(idx)

    # Plot confusion matrix for each folder
    for folder, indices in folder_to_indices.items():
        valid_indices = [i for i in indices if i < len(y_true)]
        y_true_folder = [y_true[i] for i in valid_indices]
        y_pred_folder = [y_pred[i] for i in valid_indices]

        # Ensure all classes are included in the confusion matrix
        cm = confusion_matrix(y_true_folder, y_pred_folder, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for Folder: {folder}")
        plt.savefig(f"Confusion Matrix for Folder: {folder}.png") 
        plt.close()

file_paths = test_df["file_path"].tolist()

print(f"Length of y_true_test: {len(y_true_test)}")
print(f"Length of y_pred_test: {len(y_pred_test)}")
print(f"Length of file_paths: {len(file_paths)}")

assert len(y_true_test) == len(y_pred_test) == len(file_paths), \
    "Lengths of y_true_test, y_pred_test, and file_paths must match!"

plot_confusion_matrices_by_folder(y_true_test, y_pred_test, file_paths, labels=["Real", "Fake"])


