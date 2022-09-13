"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn():
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    
    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    loss = nn.MSELoss()
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net()

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch_X, batch_Y):
        # img_a, patch_a, patch_b, corners, gt = batch
        pred = self.model(batch_X)
        lossFn = LossFn()
        loss = lossFn(pred, batch_Y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch_X, batch_Y):
        # img_a, patch_a, patch_b, corners, gt = batch
        pred = self.model(batch_X)
        lossFn = LossFn()
        loss = lossFn(pred,batch_Y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

# class Linear2D(nn.Linear):
#     def __init__(self, in_features: int, out_features: tuple, bias: bool = True, device=None, dtype=None) -> None:
#         self.out_features_shape = out_features
#         self.out_features = np.prod(self.out_features_shape)
#         super().__init__(in_features, out_features, bias, device, dtype)
        
#     def forward(self, X) :
#         return super().forward(X).reshape(self.out_features_shape)

class Net(nn.Module):
    def __init__(self, InputSize=(128,128,6), OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        # self.img_input_size = (128,128,6)
        self.homographyNet = nn.Sequential(
            
            nn.Conv2d(in_channels = 6,out_channels=64,kernel_size=(3,3),stride=1, padding=1),  #128 128 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1),#1248 128 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),#64 64 64

            nn.Conv2d(in_channels = 64,out_channels=64,kernel_size=(3,3),stride=1,padding=1), #64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),#64 64 128
            nn.BatchNorm2d(128),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #32 32 128

            nn.Conv2d(in_channels = 128,out_channels=128,kernel_size=(3,3),stride=1,padding=1),#32 32 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=1),#32 32 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),#16 16 128

            nn.Conv2d(in_channels = 128,out_channels=128,kernel_size=(3,3),stride=1,padding=1),#16 16 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=1),#16 16 128
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Dropout(p=0.5),

            nn.Flatten(),
            nn.Linear(in_features=16*16*128, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=8)
        )

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
    #     self.localization = nn.Sequential(
    #         nn.Conv2d(1, 8, kernel_size=7),
    #         nn.MaxPool2d(2, stride=2),
    #         nn.ReLU(True),
    #         nn.Conv2d(8, 10, kernel_size=5),
    #         nn.MaxPool2d(2, stride=2),
    #         nn.ReLU(True),
    #     )

    #     # Regressor for the 3 * 2 affine matrix
    #     self.fc_loc = nn.Sequential(
    #         nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
    #     )

    #     # Initialize the weights/bias with identity transformation
    #     self.fc_loc[2].weight.data.zero_()
    #     self.fc_loc[2].bias.data.copy_(
    #         torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
    #     )

    # #############################
    # # You will need to change the input size and output
    # # size for your Spatial transformer network layer!
    # #############################
    # def stn(self, x):
    #     "Spatial transformer network forward function"
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)

    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)

    #     return x

    def forward(self, X):
        """
        Input:
        X is a MiniBatch of size 128*128*2 (both images stacked)
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        out = self.homographyNet(X)
        # out = nn.LogSoftmax(dim=1)
        return out
