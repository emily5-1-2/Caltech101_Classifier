import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from Caltech101_Classifier import Caltech101_Classifier
from sklearn.model_selection import train_test_split
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)

#Fix the CLIP model so that its weights are not updated during training
for param in CLIP_model.parameters():
    param.requires_grad = False

#Download Caltech101 and transform using the preprocess function of CLIP
dataset = datasets.Caltech101(root="/Users/emilygu/Desktop", transform=CLIP_preprocess, download=True)

#Split dataset into training and validation sets
train_dset, test_dset = train_test_split(dataset, train_size=0.8)
train_dset, val_dset = train_test_split(train_dset, train_size=0.9)

train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)

wandb_logger = WandbLogger(project="Caltech101_Classifier", log_model="all")

model = Caltech101_Classifier(CLIP_model=CLIP_model, embed_dim=512, mlp_dim=200, num_classes=101)
model = model.to(device)

trainer = L.Trainer(limit_train_batches=50, max_epochs=1, default_root_dir="/Users/emilygu/Desktop/", logger=wandb_logger)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model=model, dataloaders=test_dataloader)