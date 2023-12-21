#Caltech101_Classifier Model
import lightning as L
import torch.nn as nn
import torch.optim as optim

class Caltech101_Classifier(L.LightningModule):

    def __init__(self, CLIP_model, embed_dim, mlp_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.CLIP_model = CLIP_model
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = nn.functional.cross_entropy(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def forward(self,x):
        #Embed the images through CLIP
        x = self.CLIP_model.encode_image(x)
        x = x.reshape(-1, self.embed_dim)
        #Fully connected layers for classification
        x = self.mlp(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer