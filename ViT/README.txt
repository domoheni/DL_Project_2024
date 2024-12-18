
This folder contains .py files related to the ViT Base 16x16 Google's model.

extraction_csv_70_30.py: This script is used to split the dataset into a 70% training set 
and a 30% test set while maintaining the respective ratio from each generator. It also produces
 dataframes containing 30% of the fake images from each folder, which are used for model evaluation
 in the test set after fine-tuning.

ViT_Fine_tuning.py: This script fine-tunes the model, tests it, and saves the results.

Mine_Pretrained_ViT.py: This script was used to evaluate the model on different generators.