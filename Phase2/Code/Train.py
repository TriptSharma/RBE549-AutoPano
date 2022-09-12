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
from Network.Network import HomographyModel, LossFn
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

        return np.concatenate([patch, patch_warp], axis=2), labels

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
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
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        I1 = np.float32(cv2.imread(RandImageName))
        Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinates))

    return torch.stack(I1Batch), torch.stack(CoordinatesBatch)

def GetBatch(ImgPaths, MiniBatchSize, cuda=True):
    """
    Inputs:
    ImgPaths - Paths to images
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
        RandIdx = random.randint(0, len(ImgPaths) - 1)

        # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        # RandImageName = ImgPaths[RandIdx]
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        img = cv2.imread(ImgPaths[RandIdx]).astype(np.float32)

        img_pair, Coordinates = PreprocessImage(img)
        img_pair /= 255.0
        img_pair = np.swapaxes(img_pair, 2, 0)
        # Coordinates = TrainCoordinates[RandIdx]

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(img_pair).float())
        CoordinatesBatch.append(torch.tensor(Coordinates).float())

    if cuda:
        return torch.stack(I1Batch).cuda(), torch.stack(CoordinatesBatch).cuda()
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
    TrainImagesFolder,
    ValImagesFolder,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
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

    # # TODO: add filepath to load and auto generate if no numpy file available
    # images = np.loda('./Code/training_data.npy')
    # labels = np.loda('./Code/training_labels.npy')
    #
    # np.random.seed(549)
    # shuffle_seq = np.random.permutation(images.shape[0])
    # train_split = .8
    #
    # np.reshape()
    #
    # split_idx = int(images.shape[0] * train_split)
    # images_train = images[shuffle_seq[:split_idx]]
    # labels_train = labels[shuffle_seq[:split_idx]]
    #
    # images_val = images[shuffle_seq[split_idx:]]
    # labels_val = labels[shuffle_seq[split_idx:]]
    # del images
    # del labels
    #
    # images_train = torch.tensor(images_train.reshape((-1, *images_train.shape[2:])))
    # labels_train = labels_train.reshape((-1, *labels_train.shape[2:]))
    #
    # images_val = images_val.reshape((-1, *images_val.shape[2:]))
    # labels_val = labels_val.reshape((-1, *labels_val.shape[2:]))

    train_paths = glob.glob(os.path.join(TrainImagesFolder, '*.jpg'))
    val_paths = glob.glob(os.path.join(ValImagesFolder, '*.jpg'))
    np.random.seed(549)
    np.random.shuffle(train_paths)
    np.random.shuffle(val_paths)

    NumTrainSamples = len(train_paths)
    NumValSamples = len(val_paths)
    ValBatchSize = 64

    # Predict output with forward pass
    hparams = {'InputSize': ImageSize}
    model = HomographyModel(hparams)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.Adam(model.parameters(), lr=.001)

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
        Losses = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GetBatch(train_paths, MiniBatchSize)

            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            Losses.append(LossThisBatch)

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

            result = model.validation_step(GetBatch(val_paths, ValBatchSize), PerEpochCounter)
            # Tensorboard
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                Epochs * NumIterationsPerEpoch + PerEpochCounter,
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
                "loss": np.mean(Losses),
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
        default="./Data/",
        help="Base path of images",
    )

    Parser.add_argument(
        "--TrainImagesFolder",
        default="/Users/kskrueger/Projects/RBE549_AutoPano/Phase2/Data/Train/",
        help="Folder of Training Images",
    )

    Parser.add_argument(
        "--ValImagesFolder",
        default="/Users/kskrueger/Projects/RBE549_AutoPano/Phase2/Data/Val/",
        help="Folder of Validation Images",
    )

    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
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
        default=32,
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
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    TrainImagesFolder = Args.TrainImagesFolder
    ValImagesFolder = Args.ValImagesFolder
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
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
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
        TrainImagesFolder,
        ValImagesFolder,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
