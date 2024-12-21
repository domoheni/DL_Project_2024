import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np


# Dataset directory path
data_dir = "/dtu/blackhole/10/203248/dataset/renamed_data"  # Parent folder containing 'real' and 'fake'

# Hyperparameters
img_size = (128, 128)  # Input size for ResNet50
batch_size = 64
initial_epochs = 10  # Training before unfreezing
fine_tune_epochs = 5  # Training after unfreezing
initial_learning_rate = 1e-4
fine_tune_learning_rate = 1e-5

# Load and preprocess the dataset
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Infer labels from the first-level folder names
    label_mode='int',   # Binary classification (0 or 1)
    image_size=img_size,  # Resize images to the ResNet50 input size
    batch_size=batch_size,
    shuffle=True,
    seed=123
)

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1)
])

# Preprocess the images for ResNet50
def preprocess(images, labels):
    return preprocess_input(images), labels

full_dataset = full_dataset.map(lambda x, y: (data_augmentation(x), y))  # Augment data
full_dataset = full_dataset.map(preprocess)  # Preprocess for ResNet50

# Split the dataset into train, validation, and test sets
total_size = full_dataset.cardinality().numpy()
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Optimize dataset performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the model using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model to avoid training its weights initially
base_model.trainable = False

# Build the classification model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Pooling layer to reduce feature maps
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Flatten(),  # Flatten the pooled feature maps
    Dense(128, activation='relu'),  # Another fully connected layer
    Dropout(0.5),  # Dropout again for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model for initial training
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model (initial phase)
print("Starting initial training...")
history_initial = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=initial_epochs
)

# # Unfreeze specific layers of the ResNet50 base model and added layers
# for layer in base_model.layers[-5:]:  # Unfreeze the last 5 layers of the base model
#     layer.trainable = True


# Unfreeze specific layers of the ResNet50 base model and added layers
for layer in base_model.layers:  # Unfreeze the last 5 layers of the base model
    layer.trainable = True


# Fine-tune the model (unfreeze some layers)
print("Starting fine-tuning...")
model.compile(
    optimizer=Adam(learning_rate=fine_tune_learning_rate),  # Smaller learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tune the model
history_fine_tune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=fine_tune_epochs
)

# Combine histories
history_combined = {
    'accuracy': history_initial.history['accuracy'] + history_fine_tune.history['accuracy'],
    'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tune.history['val_accuracy'],
    'loss': history_initial.history['loss'] + history_fine_tune.history['loss'],
    'val_loss': history_initial.history['val_loss'] + history_fine_tune.history['val_loss']
}

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))

# Directory to save plots
save_dir = "/dtu/blackhole/01/203777/dl_project/tenflo"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Plot the training and validation accuracy
plt.plot(history_combined['accuracy'], label='Train Accuracy')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(save_dir, 'accuracy_plot_FT2.png')
plt.savefig(accuracy_plot_path)  # Save the accuracy plot
plt.close()  # Close the plot to free up memory

# Plot the training and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history_combined['loss'], label='Train Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(save_dir, 'loss_plot_FT2.png')
plt.savefig(loss_plot_path)  # Save the loss plot
plt.close()  # Close the plot

# Function to visualize predictions
def check_model_predictions(model, dataset, class_names, num_samples=5):
    all_images, all_labels = [], []
    for images, labels in dataset:
        all_images.append(images)
        all_labels.append(labels)
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    indices = np.random.choice(len(all_images), num_samples, replace=False)
    selected_images = all_images[indices]
    selected_labels = all_labels[indices]
    
    predictions = model.predict(selected_images)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow((selected_images[i] + 1) / 2.0)  # Rescale to [0, 1] range for visualization
        plt.axis("off")
        plt.title(f"True: {class_names[int(selected_labels[i])]}\nPred: {predictions[i][0]:.4f}")
    plt.tight_layout()
    predictions_plot_path = os.path.join(save_dir, 'predictions_plot_FT2.png')
    plt.savefig(predictions_plot_path)
    plt.close()  # Close the plot

# Define class names
class_names = ['Fake', 'Real']

# Check model predictions
check_model_predictions(model, test_dataset, class_names)
