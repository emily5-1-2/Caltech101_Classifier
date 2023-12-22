import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from Caltech101_Classifier import Caltech101_Classifier
from sklearn.model_selection import train_test_split
import clip
import wandb
    
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)

#Fix the CLIP model so that its weights are not updated during training
for param in CLIP_model.parameters():
    param.requires_grad = False

#Download Caltech101 and transform using the preprocess function of CLIP
dataset = datasets.Caltech101(root="/content/drive/MyDrive", transform=CLIP_preprocess, download=True)

#Split dataset into training and validation sets
train_dset, val_dset = train_test_split(dataset, train_size=0.85)

#Start a new background process to log data to a run
wandb.init(project='Caltech101',
    config={
        "model_name": "Caltech101_Classifier",
        "batch_size": 64,
        "mlp_dim": 200,
        "learning_rate": 1e-3,
    })

#Create dataloaders
train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

#Set up wandb logger
name = "Experiment: layer_size={mlp_dim}, lr={lr}, batch={batch_size}"
wandb_logger = WandbLogger(project="Caltech101", name=name.format(mlp_dim=200, lr=1e-3, batch_size=64), log_model="all")

#Create model based on the provided hyperparameters
model = Caltech101_Classifier(CLIP_model=CLIP_model, mlp_dim=200, lr=1e-3)
model = model.to(device)

#Train model for 10 epochs
trainer = L.Trainer(max_epochs=10, logger=wandb_logger)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

wandb.finish()