from transformers import ViTForImageClassification, Trainer, TrainingArguments
from transformers import ViTImageProcessor
import torch
from datasets import load_metric
# Import necessary libraries (remains unchanged)
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from datasets import Dataset, Image, ClassLabel
from transformers import TrainingArguments, Trainer, ViTImageProcessor, ViTForImageClassification, DefaultDataCollator
import torch
from torchvision.transforms import Compose, Normalize, RandomRotation, RandomAdjustSharpness, Resize, ToTensor
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import (  # Import various metrics from scikit-learn
    accuracy_score,  # For calculating accuracy
    roc_auc_score,  # For ROC AUC score
    confusion_matrix,  # For confusion matrix
    classification_report,  # For classification report
    f1_score  # For F1 score
)
# Import custom modules and classes
from imblearn.over_sampling import RandomOverSampler # import RandomOverSampler
import accelerate # Import the 'accelerate' module
import evaluate  # Import the 'evaluate' module
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes
from transformers import (  # Import various modules from the Transformers library
    TrainingArguments,  # For training arguments
    Trainer,  # For model training
    ViTImageProcessor,  # For processing image data with ViT models
    ViTForImageClassification,  # ViT model for image classification
    DefaultDataCollator  # For collating data in the default way
)
import torch  # Import PyTorch for deep learning
from torch.utils.data import DataLoader  # For creating data loaders
from torchvision.transforms import (  # Import image transformation functions
    CenterCrop,  # Center crop an image
    Compose,  # Compose multiple image transformations
    Normalize,  # Normalize image pixel values
    RandomRotation,  # Apply random rotation to images
    RandomResizedCrop,  # Crop and resize images randomly
    RandomHorizontalFlip,  # Apply random horizontal flip
    RandomAdjustSharpness,  # Adjust sharpness randomly
    Resize,  # Resize images
    ToTensor  # Convert images to PyTorch tensors
)
# Enable the option to load truncated images.
# This setting allows the PIL library to attempt loading images even if they are corrupted or incomplete.
from transformers import ViTForImageClassification, ViTConfig
from torchvision.transforms import Compose, Resize, RandomRotation, RandomAdjustSharpness, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize


test_csv_path = "Your/path/to/test_dataset"

# Load training and testing datasets

test_df = pd.read_csv(test_csv_path)

# Convert DataFrames to Dataset objects

test_dataset = Dataset.from_pandas(test_df).cast_column("image", Image())

# Define label mappings
labels_list = ['real', 'fake']
label2id = {label: idx for idx, label in enumerate(labels_list)}
id2label = {idx: label for idx, label in enumerate(labels_list)}

# Map labels to IDs
ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)
def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

test_dataset = test_dataset.map(map_label2id, batched=True)

# Cast label column to ClassLabel

test_dataset = test_dataset.cast_column('label', ClassLabels)

# Load the saved model
saved_model_path = "Your/path/to/save/the/model"  # Path where the model was saved
model = ViTForImageClassification.from_pretrained(saved_model_path)

# Load the processor
processor = ViTImageProcessor.from_pretrained(saved_model_path)

# Define validation transformations (same as during training)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

size = processor.size["height"]
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    normalize
])

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Apply transformations to test dataset
test_dataset.set_transform(val_transforms)

# Define the collate function
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Metric computation (accuracy)
accuracy = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
    return {"accuracy": acc_score}

# Load training arguments (optional)
args = TrainingArguments(
    output_dir="eval_logs",
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    evaluation_strategy="no",
    report_to="none"
)

# Initialize Trainer with the saved model
trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")
