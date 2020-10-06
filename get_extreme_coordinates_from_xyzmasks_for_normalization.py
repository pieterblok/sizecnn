# import the libraries
import torch
import numpy as np
import os
import cv2
import random
import time
import pycocotools
import json
from shutil import copyfile

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# use open3d for some pointcloud filtering (pip install open3d)
import open3d as o3d
from pyexcel_ods import get_data
import tifffile
import csv


if __name__ == "__main__":
    readdir = "/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/xyz_masks"
    folders = ["train", "val", "test"]

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    min_z = 9999
    max_z = 0
    counter = 0

    for k in range(len(folders)):
        curdir = os.path.join(readdir, folders[k])

        if os.path.isdir(curdir):
            all_files = os.listdir(curdir)
            xyz_images = [x for x in all_files if "xyz" in x and ".tiff" in x]
            xyz_labels = [x for x in all_files if "xyz" in x and ".txt" in x]
            xyz_images.sort()
            xyz_labels.sort()


        for i in range(len(xyz_images)):
            xyzimgname = xyz_images[i]
            print(xyzimgname)
            counter = counter+1

            ## load the xyz image
            xyz_image = tifffile.imread(os.path.join(curdir, xyzimgname))

            # extract the extreme x, y and z values for normalization
            min_x_mask = np.min(xyz_image[:,:,0])
            max_x_mask = np.max(xyz_image[:,:,0])
            min_y_mask = np.min(xyz_image[:,:,1])
            max_y_mask = np.max(xyz_image[:,:,1])
            min_z_mask = np.min(xyz_image[:,:,2])
            max_z_mask = np.max(xyz_image[:,:,2])

            if min_x_mask < min_x:
                min_x = min_x_mask
                print("min_x set to: " + str(min_x))

            if min_y_mask < min_y:
                min_y = min_y_mask
                print("min_y set to: " + str(min_y))

            if min_z_mask < min_z:
                min_z = min_z_mask
                print("min_z set to: " + str(min_z))

            if max_x_mask > max_x:
                max_x = max_x_mask
                print("max_x set to: " + str(max_x))

            if max_y_mask > max_y:
                max_y = max_y_mask
                print("max_y set to: " + str(max_y))

            if max_z_mask > max_z:
                max_z = max_z_mask
                print("max_z set to: " + str(max_z))          

print("total number of images analyzed: " + str(counter))
print("min_x: " + str(min_x))
print("max_x: " + str(max_x))
print("min_y: " + str(min_y))
print("max_y: " + str(max_y))
print("min_z: " + str(min_z))
print("max_z: " + str(max_z))