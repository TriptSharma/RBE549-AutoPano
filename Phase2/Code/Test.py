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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
from Network.Network import HomographyModel
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch


# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize


def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img


def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1)) / 255.0

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1

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



def TestOperation(ImageSize, ModelPath, TestPath, LabelsPathPred):
    """
    Inputs:
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    # model = CIFAR10Model(InputSize=3 * 32 * 32, OutputSize=10)
    hparams = {'InputSize': ImageSize}
    model = HomographyModel(hparams)


    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    model.eval()
    model.model.eval()

    val_paths = glob.glob(os.path.join(TestPath, '*.jpg'))
    np.random.seed(549)
    np.random.shuffle(val_paths)

    NumValSamples = len(val_paths)
    ValBatchSize = 64

    batch_img, batch_label = GetBatch(val_paths, 32)
    gt_label = batch_label.cpu().detach().numpy()
    result = model(batch_img).cpu().detach().numpy()

    pts = result.reshape((-1, 4, 2))
    H = cv2.findHomography(np.zeros((4, 2)), pts[0])
    im2 = batch_img.cpu().detach().numpy()[0]
    im2 = np.swapaxes(im2, 0, 2)
    im2 = im2[:, :, 3:]
    corners = np.array([[0, 0], [0, 128], [128, 128], [128, 0]])
    corners2 = corners + gt_label.reshape((-1, 4, 2))[0]
    H = cv2.findHomography(corners, corners2)[0]
    im2_2 = cv2.warpPerspective(im2, np.linalg.inv(H), (300, 300))

    print("gt", gt_label)
    print("result", result)

    diff = result-gt_label

    # OutSaveT = open(LabelsPathPred, "w")

    # for count in tqdm(range(len(val_paths))):
    #     Img, Label = TestSet[count]
    #     Img, ImgOrg = ReadImages(Img)
    #     PredT = model(Img)
    #
    #     OutSaveT.write(str(PredT) + "\n")
    # OutSaveT.close()


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if not (os.path.isfile(LabelsPathTest)):
        print("ERROR: Test Labels do not exist in " + LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, "r")
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if not (os.path.isfile(LabelsPathPred)):
        print("ERROR: Pred Labels do not exist in " + LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, "r")
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())

    return LabelTest, LabelPred


def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="/media/karterk/HDD/Classes/RBE549_CV/Data/Checkpoints/100a100model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="/media/karterk/HDD/Classes/RBE549_CV/Data/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll()

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = torch.placeholder("float", shape=(1, ImageSize[0], ImageSize[1], 3))
    LabelsPathPred = "./TxtFiles/PredOut.txt"  # Path to save predicted labels

    ModelPath = "/media/karterk/HDD/Classes/RBE549_CV/Data/Checkpoints/160a100model.ckpt"

    TestOperation((128, 128, 6), ModelPath, "/media/karterk/HDD/Classes/RBE549_CV/Data/Val/", LabelsPathPred)

    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == "__main__":
    main()
