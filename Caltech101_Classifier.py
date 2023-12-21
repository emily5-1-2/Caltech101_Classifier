#Caltech101_Classifier Model
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
import lightning as L
import torch.optim as optim

class Caltech101_Classifier(L.LightningModule):

    def __init__(self, CLIP_model, embed_dim=512, mlp_dim=200, num_classes=101, lr=1e-3):
        super().__init__()
        self.embed_dim = embed_dim
        self.CLIP_model = CLIP_model
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
            nn.Softmax(dim=1)
        )
        self.lr = lr

    def forward(self,x):
        #Embed the images through CLIP
        x = self.CLIP_model.encode_image(x)
        x = x.reshape(-1, self.embed_dim)
        #Fully connected layers for classification
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = nn.functional.cross_entropy(x, y)
        preds = torch.argmax(x, dim=1)
        acc = accuracy(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = nn.functional.cross_entropy(x, y)
        preds = torch.argmax(x, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer