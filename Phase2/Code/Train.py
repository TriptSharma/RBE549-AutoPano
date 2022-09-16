#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from Network.Network import HomographyModel, LossFn, ModelType
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def PreprocessImage(img):
    i = 0
    while True:
        # print("img", i)
        i += 1
        patch_size = 128
        x = np.random.randint(patch_size // 2, img.shape[1] - patch_size // 2)
        y = np.random.randint(patch_size // 2, img.shape[0] - patch_size // 2)

        # print(x, y)

        corners = np.array([(x - patch_size // 2, y - patch_size // 2),
                            (x - patch_size // 2, y + patch_size // 2),
                            (x + patch_size // 2, y - patch_size // 2),
                            (x + patch_size // 2, y + patch_size // 2)])

        patch = img[corners[0][1]:corners[1][1], corners[0][0]:corners[3][0]]

        corners_warp = corners + np.random.normal(0, 10, (4, 2))

        warp_H = cv2.findHomography(corners, corners_warp)[0]

        # TODO: check on the out of bounds
        img_warp = cv2.warpPerspective(img, warp_H, img.shape[:2])

        patch_warp = img_warp[corners[0][1]:corners[1][1], corners[0][0]:corners[3][0]]

        img_corners = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
        img_corners_warp = cv2.perspectiveTransform(img_corners.reshape((-1, 1, 2)).astype(float), warp_H)
        inside = np.all(
            [cv2.pointPolygonTest(img_corners_warp.astype(int), corner.astype(float), False) >= 0 for corner in corners])

        if not inside:
            # print("NOT INSIDE", [cv2.pointPolygonTest(img_corners_warp.astype(int), corner.astype(float), False) >= 0 for corner in corners])
            # cv2.imshow('patch_warp', patch_warp)
            # cv2.waitKey(0)
            continue

        if patch.shape != patch_warp.shape:
            # print("mismatch size")
            continue

        labels = (corners_warp - corners).reshape((8)).astype(float)

        return labels, np.concatenate([patch, patch_warp], axis=2), img


def GenerateBatchSupervised(BasePath, TrainData, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, TrainData.shape[0] - 1)

        # RandImageName = BasePath + os.sep + TrainData[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        img, patches, labels = TrainData[RandIdx]
        Coordinates = TrainCoordinates[RandIdx]
        
        patches = patches/255
        I1 = I1.to(device)
        I1 = I1.permute((2,0,1))
        
        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(patches).float())
        CoordinatesBatch.append(torch.tensor(Coordinates).to(device))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)

def GenerateBatchUnsupervised(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    I1Batch = []
    CoordinatesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = PreprocessImage(cv2.imread(RandImageName))
        Coordinates = TrainCoordinates[RandIdx]
        
        img1 = I1[1,0]/255
        img2 = I1[1,1]/255
        
        I1 = torch.concat((img1,img2), dim=2)
        I1 = I1.permute((2,0,1))
        I1 = I1.to(device)
        
        # Append All Images and Mask
        I1Batch.append(I1)
        CoordinatesBatch.append(torch.tensor(Coordinates).to(device))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    TrainData,
    TrainCoordinates,
    ValData,
    ValCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel(ModelType).to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = AdamW(model.parameters(), lr=0.005)
    scheduler = ExponentialLR(Optimizer, gamma=0.9)
    lossFn = LossFn(ModelType)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit())) 
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatchSupervised(
                BasePath, TrainData, TrainCoordinates, ImageSize, MiniBatchSize
            ) if ModelType==ModelType.SUPERVISED else GenerateBatchUnsupervised(
                BasePath, TrainData, TrainCoordinates, ImageSize, MiniBatchSize
            )



            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = lossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

        #update scheduler after computing loss for each epoch
        scheduler.step()

        train_X, train_Y = GenerateBatch(
                BasePath, TrainData, TrainCoordinates, ImageSize, MiniBatchSize
            )
        val_X, val_Y = GenerateBatch(
                BasePath, ValData, ValCoordinates, ImageSize, MiniBatchSize
            )
        result_val = model.validation_step(val_X,val_Y)
        result_train = model.validation_step(train_X,train_Y)
        # Tensorboard
        Writer.add_scalar(
            "Validation Loss",
            result_val["val_loss"],
            Epochs,
        )
        Writer.add_scalar(
            "Training Loss",
            result_train["val_loss"],
            Epochs,
        )
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_Uns/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=10,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs_Uns/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        TrainData,
        ValData,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        ValCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        TrainData,
        TrainCoordinates,
        ValData,
        ValCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        # ModelType,
    )


if __name__ == "__main__":
    main()