import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths
model_path = "/dtu/blackhole/10/203248/dl_project/tenflo/cifake.keras"  # Path to your saved Keras model
data_dir = "/dtu/blackhole/10/203248/dl_project/DIRE/test_data"  # Path to the dataset directory

# Load the model
loaded_model = load_model(model_path)

# Function to reset model weights
def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))

# Reset the weights of the loaded model
reset_weights(loaded_model)

# Prepare the dataset
img_size = (32, 32)  # Resize to 32x32 (if that's the model's input size)
batch_size = 32  # Define batch size for training

# Load the dataset and split it into training, validation, and testing sets
# Create a 60% training, 20% validation, and 20% testing split
full_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',  # Automatically infer labels based on folder names
    label_mode='int',  # Labels are returned as integers (0 for 'fake', 1 for 'real')
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=123  # Ensure reproducibility
)

# Calculate split sizes
total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# Use `take` and `skip` to create the splits
train_dataset = full_dataset.take(train_size)
remaining_dataset = full_dataset.skip(train_size)
val_dataset = remaining_dataset.take(val_size)
test_dataset = remaining_dataset.skip(val_size)

# Prefetch data for performance optimization
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Check the number of samples in each dataset
train_samples = sum([len(batch) for batch in train_dataset])
val_samples = sum([len(batch) for batch in val_dataset])
test_samples = sum([len(batch) for batch in test_dataset])

print(f"Training dataset size: {train_samples}")
print(f"Validation dataset size: {val_samples}")
print(f"Testing dataset size: {test_samples}")

# Compile the model
loaded_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = loaded_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model on the training, validation, and test sets
train_loss, train_accuracy = loaded_model.evaluate(train_dataset)
val_loss, val_accuracy = loaded_model.evaluate(val_dataset)
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Plot the training history (accuracy and loss over epochs)
# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Function to visualize and check model predictions on random images
def check_model_predictions(model, dataset, class_names, num_samples=5):
    """
    Randomly selects `num_samples` images from `dataset`,
    displays the images, and prints the model's predictions.
    """
    # Combine all batches in the dataset into one array
    all_images, all_labels = [], []
    for images, labels in dataset:
        all_images.append(images)
        all_labels.append(labels)
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    
    # Randomly select `num_samples` images
    indices = np.random.choice(len(all_images), num_samples, replace=False)
    selected_images = all_images[indices]
    selected_labels = all_labels[indices]
    
    # Get predictions for the selected images
    predictions = model.predict(selected_images)
    
    # Plot the selected images and print predictions
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(selected_images[i].astype("uint8"))
        plt.axis("off")
        plt.title(f"True: {class_names[int(selected_labels[i])]}\nPred: {predictions[i][0]:.4f}")
    
    plt.tight_layout()
    plt.show()

# Define class names (assuming binary classification, e.g., Fake and Real)
class_names = ['Fake', 'Real']

# Call the function using the test dataset
check_model_predictions(loaded_model, test_dataset, class_names)



