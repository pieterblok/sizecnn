import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


# use open3d for some pointcloud filtering (pip install open3d)
import open3d as o3d
from pyexcel_ods import get_data
import tifffile
from tifffile import imsave
import csv


def zeropadding(xyzf, dimension=400):
    zp = np.zeros((dimension,dimension,xyzf.shape[-1])).astype(np.float32)
    diffx = int(np.divide(dimension - xyzf.shape[0], 2))
    diffy = int(np.divide(dimension - xyzf.shape[1], 2))
    zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf.astype(np.float32)
    
    return zp


def find_closest_object(boxes, coordinates_broccoli_gt, gt_data_present):
    distances = []
    if np.logical_and(boxes.size > 0, gt_data_present):
        for h in range(len(boxes)):
            box = boxes[h]
            x_center = box[0] + ((box[2] - box[0]) / 2)
            y_center = box[1] + ((box[3] - box[1]) / 2)
            distances.append(np.linalg.norm(np.asarray(coordinates_broccoli_gt) - np.asarray((x_center, y_center))))

        idx = np.asarray(distances).argmin()
    else:
        idx = []

    return idx


def scale_image(img1, max_width, max_height, interpolation_method = cv2.INTER_LINEAR):
    height, width = img1.shape[:2]
    if max_height < height or max_width < width: # only shrink if img is bigger than required
        scaling_factor = max_height / float(height) # get scaling factor
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        img1 = cv2.resize(img1, None, fx=scaling_factor, fy=scaling_factor, interpolation=interpolation_method) # resize image
    return img1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    subfolders = ["train", "val", "test"]
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    min_z = 9999
    max_z = 0

    rootdir = "./datasets/train_val_test_files"
    annotdir_visible_masks = "./datasets/annotations/mask_annotations"
    annotdir_amodal_masks = "./datasets/annotations/circle_annotations"
    gtfile = os.path.join(rootdir, "groundtruth_measurements_broccoli.ods")

    try:
        gt = get_data(gtfile)
        gt_file_present = True
    except:
        gt_file_present = False


    for j in range(len(subfolders)):
        # Initialize the image directories
        subfolder = subfolders[j]
        xyzimgdir = os.path.join(rootdir, "regression", subfolder)
        writedir = os.path.join(rootdir, "xyz_masks", subfolder)
        imgdir = os.path.join(rootdir, "mrcnn", subfolder)

        # Load the amodal dataset
        register_coco_instances("broccoli_amodal_" + subfolder, {}, os.path.join(imgdir, "annotations.json"), imgdir)
        broccoli_amodal_metadata = MetadataCatalog.get("broccoli_amodal_" + subfolder)
        dataset_dicts = DatasetCatalog.get("broccoli_amodal_" + subfolder)

        # Initialize ORCNN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)
        cfg.OUTPUT_DIR = "weights/20201010_broccoli_amodal_visible"
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0007999.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
        cfg.DATASETS.TEST = ("broccoli_amodal_" + subfolder,)
        predictor = DefaultPredictor(cfg)

        for i in range(len(dataset_dicts)):
            # Load the RGB image
            imgname = dataset_dicts[i]["file_name"]
            basename = os.path.basename(imgname)
            print(basename)
            img = cv2.imread(imgname)


            # Load the XYZ image
            xyzimgname = basename.replace("rgb", "xyz")
            xyzimgname = xyzimgname.replace(".png", ".tiff")
            xyzimg = tifffile.imread(os.path.join(xyzimgdir, xyzimgname))

            
            # Do the image inference and extract the outputs from Mask R-CNN
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()


            # Procedure to check whether we are dealing with ORCNN or MRCNN
            if "pred_visible_masks" in instances._fields:
                amodal_masks = instances.pred_masks.numpy()
                visible_masks = instances.pred_visible_masks.numpy()
            else:
                visible_masks = instances.pred_masks.numpy()


            # Get the ground truth data
            real_diameter = 0
            gt_data_present = False
            
            if gt_file_present:
                for k in range(1, len(gt['groundtruth_measurements_broccoli'])):
                    gt_data = gt['groundtruth_measurements_broccoli'][k]
                    if gt_data:
                        if gt_data[0] == basename:
                            gt_data_present = True
                            plant_id = gt_data[1]
                            real_diameter = gt_data[2]
                            x_center_gt = gt_data[3]
                            y_center_gt = gt_data[4]
                            coordinates_broccoli_gt = (x_center_gt, y_center_gt)
                        


            # If there is not a detection, use the annotation
            if boxes.size == 0:
                height, width = img.shape[:2]
                annotation = basename.replace(".png", ".json")

                # Count the number of unique annotations
                with open(os.path.join(annotdir_amodal_masks, annotation), 'r') as amodal_json:
                    data = json.load(amodal_json)
                    num_amodal_annotations = len(data['shapes'])

                amodal_masks = np.zeros((height, width, num_amodal_annotations)).astype(np.uint8)

                # Load the amodal mask from the annotation file (circle)
                with open(os.path.join(annotdir_amodal_masks, annotation), 'r') as amodal_json:
                    data = json.load(amodal_json)
                    counter = 0    
                    for p in data['shapes']:
                        cur_img = np.zeros((height, width)).astype(np.uint8)
                        xs = []
                        ys = []
                        for k in range(len(p['points'])):
                            x = p['points'][k][0]
                            y = p['points'][k][1]
                            
                            xs.append(x)
                            ys.append(y)

                        cx = xs[0]
                        cy = ys[0]
                        cx1 = xs[1]
                        cy1 = ys[1]

                        centerpoint = (int(cx), int(cy))
                        radius = int(np.sqrt(np.square(cx1-cx) + np.square(cy1-cy)))

                        cur_mask = cv2.circle(cur_img, centerpoint, radius, (255), -1)

                        amodal_masks[:,:,counter] = cur_mask
                        counter = counter+1

                # Construct a bounding box from the amodal mask
                boxes = np.zeros((num_amodal_annotations, 4)).astype(np.float32)
                for u in range(amodal_masks.shape[-1]):
                    cur_amodal_mask = amodal_masks[:,:,u]
                    ret,thresh = cv2.threshold(cur_amodal_mask,127,255,0)
                    contours,hierarchy = cv2.findContours(thresh, 1, 2)
                    cnt = contours[0]
                    x,y,w,h = cv2.boundingRect(cnt)
                    nested_array = [x, y, (x+w), (y+h)]
                    cur_bbox = np.asarray(nested_array).astype(np.float32)
                    boxes[u] = cur_bbox

                aidx = find_closest_object(boxes, coordinates_broccoli_gt, gt_data_present)
                bbox = boxes[aidx]
                amodal_mask= np.expand_dims(amodal_masks[:,:,aidx], axis=2)
                amodal_mask = np.minimum(amodal_mask, 1).astype(np.bool)

                # Count the number of unique visible annotations (polygon)
                group_ids = []  
                with open(os.path.join(annotdir_visible_masks, annotation), 'r') as visible_json:
                    data = json.load(visible_json)
                    for p in data['shapes']:
                        group_ids.append(p['group_id'])

                uniqueset = list(set(group_ids))
                group_id_numbers = list(filter(None, uniqueset))
                group_ids.count(None)
                num_visible_annotations = group_ids.count(None) + len(group_id_numbers)
                visible_masks = np.zeros((height, width, num_visible_annotations)).astype(np.uint8)

                sequence = [None] * group_ids.count(None)
                for k in range(len(group_id_numbers)):
                    group_id_number = group_id_numbers[k]
                    sequence.append(group_id_number)

                # Load the visible mask from the annotation file (polygon)
                with open(os.path.join(annotdir_visible_masks, annotation), 'r') as visible_json:
                    data = json.load(visible_json)
                    counter = 0  
                    for p in data['shapes']:
                        if p['group_id'] is None:
                            cur_img = np.zeros((height, width)).astype(np.uint8)

                            pts = []
                            for k in range(len(p['points'])):
                                pts.append(p['points'][k])

                            points = np.array(pts).astype(np.int32)

                            cur_mask = cv2.fillPoly(cur_img, [points], [255], lineType=cv2.LINE_AA)
                            visible_masks[:,:,counter] = cur_mask
                            counter = counter+1
                        else:
                            cur_img = np.zeros((height, width)).astype(np.uint8)
                            cur_group_id = p['group_id']
                            for r in data['shapes']:
                                if r['group_id'] == cur_group_id:
                                    pts = []
                                    for m in range(len(r['points'])):
                                        pts.append(r['points'][m])

                                    points = np.array(pts).astype(np.int32)

                                    cur_img = cv2.fillPoly(cur_img, [points], [255], lineType=cv2.LINE_AA)

                            fill_location = np.where(np.asarray(sequence) == cur_group_id)
                            visible_masks[:,:,int(fill_location[0])] = cur_img

                # Construct a bounding box from the visible mask
                visible_boxes = np.zeros((num_visible_annotations, 4)).astype(np.float32)
                for n in range(visible_masks.shape[-1]):
                    cur_visible_mask = visible_masks[:,:,n]
                    ret,thresh = cv2.threshold(cur_visible_mask,127,255,0)
                    contours,hierarchy = cv2.findContours(thresh, 1, 2)
                    cnt = np.concatenate(contours)
                    x,y,w,h = cv2.boundingRect(cnt)
                    nested_array = [x, y, (x+w), (y+h)]
                    cur_bbox = np.asarray(nested_array).astype(np.float32)
                    visible_boxes[n] = cur_bbox

                vidx = find_closest_object(visible_boxes, coordinates_broccoli_gt, gt_data_present)
                visible_mask= np.expand_dims(visible_masks[:,:,vidx], axis=2)
                visible_mask = np.minimum(visible_mask, 1).astype(np.bool)

            # Get the masks of the detected broccoli head that belongs to the ground truth data
            else:
                idx = find_closest_object(boxes, coordinates_broccoli_gt, gt_data_present)
                bbox = boxes[idx]
                amodal_mask = np.expand_dims(amodal_masks[idx], axis=2)
                visible_mask = np.expand_dims(visible_masks[idx], axis=2)


            # Get the XYZ data of the visible mask
            xyzimg_clip = xyzimg[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
            visible_mask_clip = visible_mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
            xyz_mask = np.multiply(xyzimg_clip, visible_mask_clip)


            # Make a final 4-channel image with the XYZ data of the visible mask and the binary amodal mask
            xyzf = np.zeros((xyz_mask.shape[0], xyz_mask.shape[1], 4), dtype=np.float32) 
            xyzf[:,:,:3] = xyz_mask.astype(np.float32)
            amodal_mask_clip = amodal_mask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
            xyzf[:,:,3] = amodal_mask_clip.reshape(xyz_mask.shape[0], xyz_mask.shape[1]).astype(np.float32)


            # Apply zeropadding to resize the final mask to a fixed size
            zp = zeropadding(xyzf, 600)


            # Write the final xyz mask and its label
            tifffile.imsave(os.path.join(writedir, xyzimgname), zp)
            txt_name = xyzimgname.replace(".tiff", ".txt")
            txtfile = open(os.path.join(writedir,txt_name),"w")
            txtfile.write("{0:.1f}".format(real_diameter))
            txtfile.close() 
            

            # Visualization
            cv2.namedWindow("XYZ Mask")
            cv2.moveWindow("XYZ Mask", 0, 0)
            cv2.imshow("XYZ Mask", zp[:,:,:3].astype(np.uint8))

            cv2.namedWindow("Amodal Mask")
            cv2.moveWindow("Amodal Mask", 700, 0)
            cv2.imshow("Amodal Mask", np.multiply(zp[:,:,3], 255).astype(np.uint8))
            k = cv2.waitKey(1)
        

            # Get the extreme x, y and z values for image normalization
            min_x_mask = np.min(zp[:,:,0])
            max_x_mask = np.max(zp[:,:,0])
            min_y_mask = np.min(zp[:,:,1])
            max_y_mask = np.max(zp[:,:,1])
            min_z_mask = np.min(zp[:,:,2])
            max_z_mask = np.max(zp[:,:,2])

            if min_x_mask < min_x:
                min_x = min_x_mask

            if min_y_mask < min_y:
                min_y = min_y_mask

            if min_z_mask < min_z:
                min_z = min_z_mask

            if max_x_mask > max_x:
                max_x = max_x_mask

            if max_y_mask > max_y:
                max_y = max_y_mask

            if max_z_mask > max_z:
                max_z = max_z_mask


print("min_x: " + str(min_x))
print("max_x: " + str(max_x))
print("min_y: " + str(min_y))
print("max_y: " + str(max_y))
print("min_z: " + str(min_z))
print("max_z: " + str(max_z))