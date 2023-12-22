# Caltech101_Classifier

This is a classifier for the Caltech 101 image dataset. The model makes use of image embeddings obtained from CLIP, which are then input into a two-layer MLP for classification. The PyTorch Lightning framework was used.

* Caltech101_Classifier.py defines the model
* train_classifier.py shows sample code for preparing the Caltech 101 dataset and training the classifier. Results are logged using wandb
* Caltech101_Classifier_Colab.ipynb is a Colab notebook that combines the code from Caltech101_Classifier.py and train_classifier.py. Simple experiments for changing hyperparameters were done. The results are shown in the following links:
    - Layer Size: https://api.wandb.ai/links/emilygu/q09uxf9q
    - Learning Rate: https://api.wandb.ai/links/emilygu/iamm0mgw
    - Batch Size: https://api.wandb.ai/links/emilygu/kn5o2zc9

