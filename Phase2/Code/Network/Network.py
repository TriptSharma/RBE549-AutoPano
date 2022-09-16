"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

from turtle import forward
from webbrowser import Konqueror
import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl
from enum import Enum

# Don't generate pyc codes
sys.dont_write_bytecode = True

class ModelType(Enum):
    SUPERVISED = 0
    UNSUPERVISED = 1

def LossFn(modelType):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    
    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    loss = nn.L1Loss() if modelType==ModelType.UNSUPERVISED else nn.MSELoss()
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self, ModelType):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.modelType = ModelType
        self.model = Net(self.modelType)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch_X, batch_Y):
        # img_a, patch_a, patch_b, corners, gt = batch
        pred = self.model(batch_X)
        lossFn = LossFn(self.modelType)
        loss = lossFn(pred, batch_Y)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch_X, batch_Y):
        # img_a, patch_a, patch_b, corners, gt = batch
        pred = self.model(batch_X)
        lossFn = LossFn(self.modelType)
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

class TensorDLT(nn.Module):
    def __init__(self) -> None:
        '''
        H_4pt_X is H_4pt for batch X
        H_4pt_ is the predicted H_4pt by the homography net, use it to calculate the corners in the predicted/ warped image
        C_a are the corner points of the patch in Image A or in this case the training image
        '''
        # self.H_4pt_ = H_4pt_
        # self.C_a = C_a
        super().__init__()

    def tensorDLT(self, H_4pt_X, C_a):
        H = torch.tensor([])
        for H_4pt_ in H_4pt_X:
            #corners are [x1 y1 x2 y2 x3 y3 x4 y4]
            C_b_ = C_a + H_4pt_
            A = []
            b = []
            for i in range(0,8,2): #since there are 4 corner pairs
                Ai = [[0, 0, 0, -C_a[i], -C_a[i+1], -1, C_b_[i+1]*C_a[i], C_b_[i+1]*C_a[i+1]]]
                Ai.append([C_a[i], C_a[i+1], 1, 0, 0, 0, -C_b_[i]*C_a[i], -C_b_[i]*C_a[i+1]])
                A.extend(Ai)
            
                bi = [-C_b_[i+1],-C_b_[i]]
                b.extend(bi)

            A = torch.tensor(A).to('cuda')
            b = torch.tensor(b).to('cuda')
            # h = inv(A) dot b
            print(A)
            h = torch.dot(torch.inverse(A), b)
            H = torch.cat(H,h.reshape(1,-1), axis=0)
        H = H[1:,:]
        print(H.shape)
        return H

    def forward(self,H_4pt_X, C_a):
        return self.tensorDLT(H_4pt_X, C_a)

class STNLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, X, H):
        return kornia.geometry.transform.warp_perspective(X,H)


class Net(nn.Module):
    def __init__(self, ModelType, InputSize=(128,128,6), OutputSize=8):
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
        self.model_type = ModelType
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
        # Adding Tensor DLT layer
        #############################
        self.tensorDLT = TensorDLT()
        self.stnLayer = STNLayer()
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), 
            nn.ReLU(), 
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    def stn(self, x):
        "Spatial transformer network forward function"
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

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
        #homography return c_4pt
        h_4pt = out
        if(self.model_type==ModelType.UNSUPERVISED):
            C_a = torch.tensor([0,0,0,X[0].shape[1],0,X[0].shape[2],X[0].shape[1],X[0].shape[2]]).to('cuda')
            out = self.tensorDLT(h_4pt, C_a) # B x 8
            #append 1 to the each h vector and reshape
            out = torch.cat((out,torch.ones((out.size(dim=0),1))),dim=1)
            out = out.reshape((out.size(dim=0),3,3))
            # print(out.size())
            out = self.stnLayer(X, out)
            print('unsupervised works XD')

        return out