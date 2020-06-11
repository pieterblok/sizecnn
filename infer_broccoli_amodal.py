# import the libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch
import numpy as np
import os
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'


def load_rgb_image(imagepath):
    try:
        img = cv2.imread(imagepath)
    except (FileNotFoundError):
        print("cannot load the rgb image... close the program")
        exit(1)
    return img


def load_xyz_image(imagepath):
    try:
        xyzimg = cv2.imread(imagepath,-1)

        if len(xyzimg.shape) == 3:
            # be aware opencv2 reads an image in reversed order (so RGB->BGR and XYZ->ZYX)
            xyzimg = xyzimg[...,::-1]

    except (FileNotFoundError):
        print("cannot load the xyz image... close the program")
        exit(1)
    return xyzimg


# do not use this function in the final code (it's not very fast) but use it purely to visualize the intermediate steps for debugging
def visualize(img, amodal_masks, modal_masks):
    amodal_masks = amodal_masks.astype(dtype=np.uint8)
    modal_masks = modal_masks.astype(dtype=np.uint8)
    max_height = 900
    max_width = 900

    if amodal_masks.any() and modal_masks.any():
        amodal_maskstransposed = amodal_masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        modal_maskstransposed = modal_masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        
        red_mask = np.zeros((amodal_maskstransposed.shape[0],amodal_maskstransposed.shape[1]),dtype=np.uint8)
        green_mask = np.zeros((amodal_maskstransposed.shape[0],amodal_maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((amodal_maskstransposed.shape[0],amodal_maskstransposed.shape[1],3),dtype=np.uint8) # BGR

        for i in range (modal_maskstransposed.shape[-1]):
            modal_mask = modal_maskstransposed[:,:,i]
            green_mask = cv2.add(green_mask,modal_mask)

        for j in range (amodal_maskstransposed.shape[-1]):
            amodal_mask = amodal_maskstransposed[:,:,j]
            mask_diff = np.subtract(amodal_mask,green_mask)
            red_mask = cv2.add(red_mask,mask_diff)

        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        height, width = img.shape[:2]
        img_mask = cv2.addWeighted(img,1,all_masks,0.5,0)

        if max_height < height or max_width < width: # only shrink if img is bigger than required
            scaling_factor = max_height / float(height) # get scaling factor
            if max_width/float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

        cv2.imshow("RGB image with mask(s)", img_mask) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
        cv2.waitKey(0)
    else:
        height, width = img.shape[:2]
        
        if max_height < height or max_width < width: # only shrink if img is bigger than required
            scaling_factor = max_height / float(height) # get scaling factor
            if max_width/float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

        cv2.imshow("RGB image", img) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
        cv2.waitKey(0)

    cv2.destroyAllWindows()

  
if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    if torch.cuda.is_available():
        # configure the inference procedure
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TEST = ("broccoli_amodal_test",)
        cfg.NUM_GPUS = 2
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)
        cfg.OUTPUT_DIR = "weights/broccoli_amodal"
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

        # initialize the predictor (for image inference)
        predictor = DefaultPredictor(cfg)

        # read rgb and xyz image
        imgdir = '/media/pieterdeeplearn/easystore/BroccoliData/20190620'
        rgbimgname = '20190620_182616688_RGB_2.tif'
        xyzimgname = '20190620_182616688_Depth_2.tif'

        img = load_rgb_image(os.path.join(imgdir,rgbimgname))
        xyzimg = load_xyz_image(os.path.join(imgdir,xyzimgname))

        # infer the predictor on the rgb image
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        bbox = instances.pred_boxes.tensor.numpy()
        amodal_masks = instances.pred_masks.numpy()
        modal_masks = instances.pred_visible_masks.numpy()
        invisible_masks = instances.pred_invisible_masks.numpy()

        # visualize the rgb img with the amodal masks in red and the modal masks in green
        visualize(img, amodal_masks, modal_masks)

        