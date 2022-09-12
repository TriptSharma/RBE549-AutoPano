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


# def LossFn(delta, img_a, patch_b, corners):
def LossFn(delta, gt_delta):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################

    # loss = ...
    loss = nn.MSELoss()(delta, gt_delta)
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self, hparams):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = Net(hparams['InputSize'], None)

    # def forward(self, a, b):
    #     return self.model(a, b)
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx=-1):
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(patch_a, patch_b)
        # loss = LossFn(delta, img_a, patch_b, corners)
        imgs, labels = batch
        delta = self.model(imgs)
        loss = LossFn(delta, labels)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx=-1):
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(patch_a, patch_b)
        # loss = LossFn(delta, img_a, patch_b, corners)
        imgs, labels = batch
        delta = self.model(imgs)
        loss = LossFn(delta, labels)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class Net(nn.Module):
    def __init__(self, InputSize=(128, 128, 6), OutputSize=(8,)):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        self.input_size = InputSize  # HxWxC

        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################

        def Conv_Block(input_c, output_c, kernel, stride, padding=None):
            if padding is None:
                padding = kernel // 2
            return nn.Sequential(nn.Conv2d(input_c, output_c, kernel, stride, padding=padding),
                                 nn.BatchNorm2d(output_c),
                                 nn.ReLU())


        self.net = nn.Sequential(
            Conv_Block(self.input_size[2], 64, 3, 1, padding=3//2),
            Conv_Block(64, 64, 3, 1, padding=3//2),
            nn.MaxPool2d(2, 2),
            Conv_Block(64, 64, 3, 1, padding=3//2),
            Conv_Block(64, 64, 3, 1, padding=3//2),
            nn.MaxPool2d(2, 2),
            Conv_Block(64, 128, 3, 1, padding=3//2),
            Conv_Block(128, 128, 3, 1, padding=3//2),
            nn.MaxPool2d(2, 2),
            Conv_Block(128, 128, 3, 1, padding=3//2),
            nn.Dropout2d(.5),  # TODO: "Dropout applied after final 2 conv layers" is this extra?
            Conv_Block(128, 128, 3, 1, padding=3//2),
            nn.Dropout2d(.5),
            nn.Flatten(),
            nn.Linear(16*16*128, 1024),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(1024, 8),
            # nn.Linear(1024, 8*21),
        )

        self.net.float()
        self.net.cuda()

        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # unsupervised = False
        # if unsupervised:
        #     # Spatial transformer localization-network
        #     self.localization = nn.Sequential(
        #         nn.Conv2d(1, 8, kernel_size=7),
        #         nn.MaxPool2d(2, stride=2),
        #         nn.ReLU(True),
        #         nn.Conv2d(8, 10, kernel_size=5),
        #         nn.MaxPool2d(2, stride=2),
        #         nn.ReLU(True),
        #     )
        #
        #     # Regressor for the 3 * 2 affine matrix
        #     self.fc_loc = nn.Sequential(
        #         nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        #     )
        #
        #     # Initialize the weights/bias with identity transformation
        #     self.fc_loc[2].weight.data.zero_()
        #     self.fc_loc[2].bias.data.copy_(
        #         torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        #     )

    #############################
    # You will need to change the input size and output
    # size for your Spatial transformer network layer!
    #############################
    # def stn(self, x):
    #     "Spatial transformer network forward function"
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #
    #     return x

    # def forward(self, xa, xb):
    def forward(self, x_input):
        """
        Input:
        x_input is a MiniBatch of the images (a, b) stacked as 6 channels
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        # x_input = torch.concat([xa, xb], axis=1)  # Axis 1 assumes BxCxHxW axis order
        # self.net.float()
        x_net = self.net(x_input)
        # x_net = x_net.view((-1, 8))
        # out = nn.Softmax(dim=2)(x_net)  # TODO: dim=2 along the 21 "classes" gets output Bx8?
        out = x_net

        return out
