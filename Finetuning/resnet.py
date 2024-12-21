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


# Dataset directory path
data_dir = "/dtu/blackhole/10/203248/dataset/renamed_data"  # Parent folder containing 'real' and 'fake'

# Hyperparameters
img_size = (128, 128)  # Input size
batch_size = 64
learning_rate = 1e-4

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

# Preprocess the images for ResNet50
def preprocess(images, labels):
    return preprocess_input(images), labels

full_dataset = full_dataset.map(preprocess)

# Split the dataset into train, validation, and test sets
total_size = tf.data.experimental.cardinality(full_dataset).numpy() * batch_size
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

train_dataset = full_dataset.take(train_size // batch_size)
remaining_dataset = full_dataset.skip(train_size // batch_size)
val_dataset = remaining_dataset.take(val_size // batch_size)
test_dataset = remaining_dataset.skip(val_size // batch_size)

# Optimize dataset performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the model using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model to avoid training its weights
base_model.trainable = False

# Build the classification model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Flatten(),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
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
print(f"Testing Accuracy (Untrained Model): {test_accuracy:.4f}")