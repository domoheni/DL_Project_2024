import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D, RandomFlip, RandomRotation
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os



# Path to the dataset directory
data_dir = "/dtu/blackhole/10/203248/dataset/renamed_data"

# Prepare the dataset
# Hyperparameters
img_size = (128, 128)  # Input size for ResNet50
batch_size = 64
epochs = 10
learning_rate = 1e-4

full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Automatically infer labels based on folder names
    label_mode='int',   # Binary classification (0 or 1)
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=123
)

# Apply preprocessing to match ResNet50 requirements
def preprocess(images, labels):
    return preprocess_input(images), labels

full_dataset = full_dataset.map(preprocess)

# Splitting the dataset
total_size = full_dataset.cardinality().numpy()
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Prefetch data for performance optimization
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model to retain pre-trained features
base_model.trainable = True

# Build the classification model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Pooling layer to reduce feature maps
    Flatten(),  # Fully connected layer
    Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 5))

# Directory to save plots
save_dir = "/dtu/blackhole/01/203777/dl_project/tenflo"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
accuracy_plot_path = os.path.join(save_dir, 'accuracy_plot_full_train2.png')
plt.savefig(accuracy_plot_path)  # Save the accuracy plot
plt.close()  # Close the plot to free up memory

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_plot_path = os.path.join(save_dir, 'loss_plot_full_train2.png')
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
    predictions_plot_path = os.path.join(save_dir, 'predictions_plot_full_train2.png')
    plt.savefig(predictions_plot_path)
    plt.close()  # Close the plot

# Define class names
class_names = ['Fake', 'Real']

# Check model predictions
check_model_predictions(model, test_dataset, class_names)
