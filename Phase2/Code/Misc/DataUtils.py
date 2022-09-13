"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import torch

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupData():
    train_data = np.load("../Data/TrainData/training_data.npy")
    train_labels = np.load("../Data/TrainData/training_labels.npy")
    # train_labels_H = np.load("../Data/TrainData/training_labels_H.npy")

    val_data = np.load("../Data/ValData/val_data.npy")
    val_labels = np.load("../Data/ValData/val_labels.npy")
    # val_labels_H = np.load("../Data/ValData/val_labels_H.npy")

    train_data = train_data.reshape((train_data.shape[0]*train_data.shape[1], train_data.shape[2], train_data.shape[3],train_data.shape[4],train_data.shape[5]))
    train_labels = train_labels.reshape((train_labels.shape[0]*train_labels.shape[1], train_labels.shape[2], train_labels.shape[3],train_labels.shape[4]))

    val_data = val_data.reshape((val_data.shape[0]*val_data.shape[1], val_data.shape[2], val_data.shape[3],val_data.shape[4],val_data.shape[5]))
    val_labels = val_labels.reshape((val_labels.shape[0]*val_labels.shape[1], val_labels.shape[2], val_labels.shape[3]*val_labels.shape[4]))

    train_labels = (train_labels[:,1]-train_labels[:,0]).reshape(-1,8)
    val_labels = (val_labels[:,1]-val_labels[:,0]).reshape(-1,8)


    train_data = torch.from_numpy(np.float32(train_data))
    val_data = torch.from_numpy(np.float32(val_data))
    train_labels = torch.from_numpy(np.float32(train_labels))
    val_labels = torch.from_numpy(np.float32(val_labels))

    print("train and val datasets loaded into RAM successfully")
    return train_data, val_data, train_labels, val_labels

def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    # DirNamesTrain = SetupDirNames(BasePath)

    # # Read and Setup Labels
    # LabelsPathTrain = '../Data/TrainData/training_labels.npy'
    # TrainLabels = ReadLabels(LabelsPathTrain)

    # LabelsPathVal = '../Data/ValData/val_labels.npy'
    # ValLabels = ReadLabels(LabelsPathVal)

    #get train and val datasets
    TrainData, ValData, TrainLabels, ValLabels = SetupData()

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 1000
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 3]
    NumTrainSamples = TrainData.shape[0]

    # Number of classes
    NumClasses = 8

    return (
        TrainData,
        ValData,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        ValLabels,
        NumClasses,
    )


def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, "r")
        TrainLabels = TrainLabels.read()
        TrainLabels = list(map(float, TrainLabels.ssplit()))
        # TrainLabels = np.load(LabelsPathTrain)

    return TrainLabels

def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames("./TxtFiles/DirNamesTrain.txt")

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
