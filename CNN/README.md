# CNN
## File Structure

This folder contains the files related to the custom-built cnn worked on for the image classification task. It constains a python file to load the data and it's respective bash file to run on DTU's HPC and the same for the Model python file. It also contains the python libraries requirements to clarify dependencies and the last outcome files seen as **deepCNN_23394188.out** and **deepCNN_23394188.err**

## Model

The custom built CNN was deeper and more robust than a previously worked on shallower CNN. The model started with three convolutional blocks, each consisting of two convolutional layers followed by batch normalization and ReLU activations. These blocks were followed by max-pooling layers for the progressive down-sampling of the input into smaller spatial dimensions, retaining essential features. The first block used 32 and 64 filters in its two layers, while the second block increased this to 128 filters in both layers. The third block further scaled up the feature extraction with 256 filters in both layers. This progression allowed the model to capture increasingly complex patterns, from simple edges and textures to intricate shapes and abstract features.

To stabilize learning and speed up convergence, batch normalization was applied after each convolutional layer. Finally, at the end of each block, max pooling with a kernel size of 2 × 2 was performed to reduce the spatial resolution, reducing overfitting and computation. The final output from the convolutional blocks would  have a size of 256×16×16, which was flattened to a feature vector of size 65,536 as input for the fully connected layers.

The fully connected block consisted of a dense layer, which projected the high-dimensional feature vector onto a lower dimensionality of 512 units, followed by a dropout layer with a 50 percent probability, thus aiding to preventing overfitting by setting the neurons to zero randomly during training. Afterwards, the feature space is further compressed by a second fully connected layer, now to 128 units, also followed by dropout. The last dense layer outputed a single value representing the probability of the input belonging to the class "fake." A sigmoid activation function was later then applied on this output, which ensured it would be in the range between 0 and 1, suitable for binary classification tasks.


## Outcome

As expected, the CNN demonstrated robust generalization and accuracy across almost all the generators present in the dataset, achieving good accuracies on validation, testing, and augmented testing: **88.261%**, **88.191%**, and **87.035%**, respectively.

For a more detailed plots and visualization. The detailed plots can be found on the __Plot Results__ folder. This folder is organized by levels where:
- Shallow level contains:
    - Per Generator Folder
    - Predictions Folder
    - Model Architecture
    - Training and Validation Loss
    - Confusion Matrices for Test and Validation

- The Deeper Levels (the folders) contain:
    - The Confusion Matrices for each generators and also the real images
    - A few examples of predictions using the model