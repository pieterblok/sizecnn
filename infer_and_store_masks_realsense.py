# import the libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
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

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.checkpoint import DetectionCheckpointer

# use open3d for some pointcloud filtering (pip install open3d)
import open3d as o3d
from pyexcel_ods import get_data
from tifffile import imsave

class ProcessImage:
    def __init__(self, model, cfg):
        self.model = model
        self.input_format = cfg.INPUT.FORMAT
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std


    def load_rgb_image(self, imagepath):
        try:
            original_image = cv2.imread(imagepath)
        except (FileNotFoundError):
            print("cannot load the rgb image... close the program")
            exit(1)
        return original_image


    def load_xyz_image(self, imagepath):
        try:
            xyzimg = cv2.imread(imagepath,-1)

            if len(xyzimg.shape) == 3:
                # be aware opencv2 reads an image in reversed order (so RGB->BGR and XYZ->ZYX)
                xyzimg = xyzimg[...,::-1]

        except (FileNotFoundError):
            print("cannot load the xyz image... close the program")
            exit(1)
        return xyzimg


    def maketensor(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            input = {"image": image, "height": height, "width": width}
            return input


    def preprocess_tensor(self, input):
        """
        Normalize, pad and batch the input images.
        """
        image = input["image"].to(self.device)
        image = [self.normalizer(image)]
        image = ImageList.from_tensors(image, self.backbone.size_divisibility)
        return image

    
    # do not use this function in the final code (it's not very fast) but use it purely to visualize the intermediate steps for debugging
    def visualize(self, img, amodal_masks, modal_masks):
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

            cv2.imshow("RGB image with amodal masks (red) and modal masks (green)", img_mask) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
            k = cv2.waitKey(0)

            if k == 27:
                cv2.destroyAllWindows()

        else:
            height, width = img.shape[:2]
            
            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

            cv2.imshow("RGB image", img) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
            k = cv2.waitKey(0)

            if k == 27:
                cv2.destroyAllWindows()


    def postprocess(self, xyzimg, masks, max_depth_range_broc=100, max_depth_contribution=0.04):
        masks = masks.astype(np.uint8)
        x = np.expand_dims(xyzimg[:,:,0], axis=2)
        y = np.expand_dims(xyzimg[:,:,1], axis=2)
        z = np.expand_dims(xyzimg[:,:,2], axis=2)

        height, width = xyzimg.shape[:2]
        z_negative = False

        # check if the z-image has positive values or negative
        if np.min(z) < 0:
            z = np.multiply(z,-1)
            z_negative = True

        if masks.any():
            md, mh, mw = masks.shape
            masks_final = np.zeros((md,mh,mw),dtype=np.uint8)
            maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
            zts = np.zeros(maskstransposed.shape[-1])
            zes = np.zeros(maskstransposed.shape[-1])
            cXs = np.zeros(maskstransposed.shape[-1],dtype=np.uint16)
            cYs = np.zeros(maskstransposed.shape[-1],dtype=np.uint16)
            diameters = np.zeros(maskstransposed.shape[-1],dtype=np.float32)

            for i in range (maskstransposed.shape[-1]):
                masksel = maskstransposed[:,:,i]
                ret,broc_mask = cv2.threshold((masksel*255).astype(np.uint8),1,255,cv2.THRESH_BINARY)
                output = cv2.connectedComponentsWithStats(broc_mask)
                num_labels = output[0] # The first cell is the number of labels
                labels = output[1] # The second cell is the label matrix
                stats = output[2] # The third cell is the stat matrix
                centroids = output[3] # The fourth cell is the centroid matrix

                region_areas = []

                for label in range(1,num_labels):
                    region_areas.append(stats[label, cv2.CC_STAT_AREA])

                # procedure to remove the very small separated broccoli masks
                if len(region_areas) > 1:
                    for w in range(len(np.asarray(region_areas))):
                        region_area = np.asarray(region_areas)[w]
                        if region_area < 100:
                            masksel = np.where(labels==(w+1), 0, masksel)                

                contours, hierarchy = cv2.findContours((masksel*255).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cnt = np.concatenate(contours)
                (x,y),radius = cv2.minEnclosingCircle(cnt)

                # black_img = np.zeros((mh,mw),dtype=np.uint8)
                # cntimg = cv2.drawContours(black_img, contours, -1, (255), 1)
                # cv2.circle(cntimg, (int(x), int(y)), 7, (255), -1)
                # cv2.circle(cntimg, (int(x), int(y)), int(radius),(255),2)
                # cv2.imshow("contours",cntimg)
                # cv2.waitKey(0)

                cXs[i] = int(x)
                cYs[i] = int(y)
                diameters[i] = radius*2

                masksel_bool = masksel.astype(np.bool) # convert the mask to a boolean mask where ones become True and zeros False
                
                z_mask = z[masksel_bool] # clip to the z-values only to the region of the mask
                z_mask_filtered = z_mask[z_mask != 0] # filter the z_mask for all values > 0

                if z_mask_filtered.size > 1: # this conditional statement is needed to only analyse the masks that have at least 1 pixel
                    z_mask_filtered_range = np.max(z_mask_filtered)-np.min(z_mask_filtered) # calculate the range of z-values inside the filtered mask

                    # this conditional statement filters the z_mask using its histogram
                    if (z_mask_filtered_range > max_depth_range_broc):
                        hist, bin_edges = np.histogram(z_mask_filtered, density=False) # create a histogram of 10 bins within the filtered broccoli mask
                        hist_peak = np.argmax(hist) # get the depth-value with the highest number of bin_counts (peak)
                        lb = bin_edges[hist_peak]
                        ub = bin_edges[hist_peak+1]

                        bc = np.bincount(np.absolute(z_mask_filtered.astype(np.int64))) # calculate the bin-count for every unique depth value
                        peak_id = np.argmax(bc) # get the depth-value with the highest number of bin_counts (peak)

                        # check if the largest peak falls within the dominant histogram peak, if not find the another peak within the dominant histogram peak
                        if peak_id > int(lb) and peak_id < int(ub):
                            peak_id = peak_id
                        else:
                            bc_clip = bc[int(lb):int(ub)]
                            peak_id = int(lb) + np.argmax(bc_clip)

                        pixel_counts = np.zeros((10), dtype=np.int64)

                        for j in range(10):
                            lower_bound = peak_id-(max_depth_range_broc - (j * 10)) # get the lower-bound (z-min) of the dominant peak
                            upper_bound = lower_bound + max_depth_range_broc # get the upper-bound (z-max) of the dominant peak
                            z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))] # select the z-values of the dominant histogram bin
                            pixel_counts[j] = z_final.size

                        pix_id = np.argmax(pixel_counts) # select the pixel_counts id with the highest number of pixels, if there are two (equal) id's select the first
                        lower_bound = peak_id-(max_depth_range_broc - (pix_id * 10))
                        upper_bound = lower_bound + max_depth_range_broc
                        z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
                        
                        hist_f, bin_edges_f = np.histogram(z_final, density=False) # create a final histogram of 10 bins within the filtered broccoli mask
                        norm1 = hist_f / np.sum(hist_f) # create a normalized histogram with a cumulative sum of 1.0

                    else:
                        hist_f, bin_edges_f = np.histogram(z_mask_filtered, density=False) # create a final histogram of 10 bins within the filtered broccoli mask
                        norm1 = hist_f / np.sum(hist_f) # create a normalized histogram with a cumulative sum of 1.0

                    # plot histogram
                    #plt.hist(z_final, 10)
                    #plt.show()

                    #plt.bar(np.arange(10), norm1)
                    #plt.show()

                    sel1 = bin_edges_f[np.where(norm1 >= max_depth_contribution)] # select the lower bound of the bins that contribute for at least 5% to the histogram depth
                    sel2 = bin_edges_f[np.where(norm1 >= max_depth_contribution)[0]+1] # select the lower bound of the bins that contribute for at least 5% to the histogram depth
                    edges = np.concatenate((sel1,sel2), axis=0) # concatenate all bins
                    final_bins = np.unique(edges) # get only the unique bins

                    #BE AWARE OF THE COORDINATE-SYSTEM: THE Z-PLANE CAN BE NEGATIVE OR POSITIVE: 
                    z_top = np.min(final_bins) # extract the z-top as minimum z-value (closest to the camera)
                    z_edge = np.max(final_bins) # extract the z-edge as maximum z-value (most remote from the camera)

                    zts[i] = z_top
                    zes[i] = z_edge
                    masks_final[i,:,:] = masksel
                else:
                    zts[i] = 0
                    zes[i] = 0
                    masks_final[i,:,:] = masksel

        else:
            z_top = 0 
            z_edge = 0
            cXs = 0
            cYs = 0

            zts = z_top
            zes = z_edge
            diameters = 0
            CPs = np.zeros(4,dtype=np.int32)

            masks_final = masks

            print("no 3D postprocessing possible, because there are no masks")

        # check if the z-image has positive values or negative
        if z_negative:
            z = np.multiply(z,-1)
            zts = np.multiply(zts,-1)
            zes = np.multiply(zes,-1)
        
        return z, zts, zes, masks_final, cXs, cYs, diameters


    def visualize_masks(self, img, z, boxes, masks, zt, ze, cXs, cYs, diameters, diametersmm, real_diameter):       
        masks = masks.astype(np.uint8)
        height, width = z.shape[:2]
        max_height = 700
        max_width = 700

        z_negative = False

        # check if the z-image has positive values or negative
        if np.min(z) < 0:
            z = np.multiply(z, -1)
            zt = np.multiply(zt, -1)
            ze = np.multiply(ze, -1)
            z_negative = True

        if masks.any():
            maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
            red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8) # BGR
            
            z_img_final = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.uint8)
            
            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks
                
                bbox = boxes[i].astype(np.uint16)
                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                zsel = z[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the z-image to the boxing box to reduce memory load

                z_mask = np.multiply(zsel,masksel) # clip to the z-values only to the region of the mask, but keep the matrix dimensions intact for visualization
                z_top = zt[i]
                z_edge = ze[i]
                z_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),z_mask,0) # this code keeps the matrix dimensions intact for visualization
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) # make the depth image binary by clipping everything with value>1 to value 1 

                z_img = z_mask_final.copy() # a z-image where the 0 values are removed
                color_range = 200 # from 0-200
                if int(z_top) != 0 and int(z_edge) != 0:
                    np.clip(z_img, z_top, z_edge, out=z_img)
                    z_img = np.interp(z_img, (z_img.min(), z_img.max()), (0, 200))
                z_img = z_img.astype(np.uint8) # make the 8-bit image of the z-values for visualization
                z_img_final[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_img

                mask_diff = np.subtract(masksel,z_mask_final_binary)
                mask_diff = mask_diff.reshape(mask_diff.shape[0],mask_diff.shape[1])
                red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff

                z_mask_final_binary = z_mask_final_binary.reshape(z_mask_final_binary.shape[0],z_mask_final_binary.shape[1])
                green_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_mask_final_binary

            all_masks[:,:,0] = blue_mask
            all_masks[:,:,1] = green_mask
            all_masks[:,:,2] = red_mask
            all_masks = np.multiply(all_masks,255).astype(np.uint8)

            z_img_final = np.where(np.absolute(z_img_final)==0, 200, z_img_final) # this is an time efficient way to do the clipping
            z_img_final = np.multiply(z_img_final, np.minimum(z,1))
            z_img_final = z_img_final.astype(np.uint8)
            z3 = cv2.cvtColor(z_img_final,cv2.COLOR_GRAY2RGB)  

            img_mask = cv2.addWeighted(img,1,all_masks,0.5,0) # overlay the RGB image with the red full mask
            zimg_mask = cv2.addWeighted(z3,1,all_masks,0.5,0) # overlay the z3 image with the red full mask

            if z_negative:
                zt = np.multiply(zt, -1)
                ze = np.multiply(ze, -1)

            for k in range(cXs.size):
                cv2.circle(img_mask, (cXs[k], cYs[k]), 7, (255, 255, 255), -1)
                cv2.circle(img_mask, (cXs[k], cYs[k]), int(diameters[k]/2),(255,0,0),3)
                cv2.putText(img_mask, "Z-top: {:.1f} mm".format(zt[k]),(cXs[k]+10,cYs[k]-15),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
                cv2.putText(img_mask, "Z-edge: {:.1f} mm".format(ze[k]),(cXs[k]+10,cYs[k]+20),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
                cv2.putText(img_mask, "Diameter: {:.1f} mm".format(diametersmm[k]),(cXs[k]+10,cYs[k]+55),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
                cv2.putText(img_mask, "Real diameter: {} mm".format(real_diameter),(cXs[k]+10,cYs[k]+90),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))

                cv2.circle(zimg_mask, (cXs[k], cYs[k]), 7, (0, 0, 0), -1)
                cv2.circle(zimg_mask, (cXs[k], cYs[k]), int(diameters[k]/2), (255,0,0), 3)
                cv2.putText(zimg_mask, "Z-top: {:.1f} mm".format(zt[k]),(cXs[k]+10,cYs[k]-15),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                cv2.putText(zimg_mask, "Z-edge: {:.1f} mm".format(ze[k]),(cXs[k]+10,cYs[k]+20),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                cv2.putText(zimg_mask, "Diameter: {:.1f} mm".format(diametersmm[k]),(cXs[k]+10,cYs[k]+55),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                cv2.putText(zimg_mask, "Real diameter: {} mm".format(real_diameter),(cXs[k]+10,cYs[k]+90),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR) # resize image
                zimg_mask = cv2.resize(zimg_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR) # resize image

            cv2.namedWindow("RGB")
            cv2.moveWindow("RGB", 0, 0)
            cv2.imshow("RGB", img_mask) # Show image

            cv2.namedWindow("Depth")
            cv2.moveWindow("Depth", max_width+100, 0)
            cv2.imshow("Depth" , zimg_mask)
            cv2.waitKey(0)
      
        else:
            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image
                z = cv2.resize(z.astype(np.uint8), None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

            z_img_binary = np.minimum(z,1)
            zimg = np.multiply(z_img_binary,200)

            cv2.namedWindow("RGB")
            cv2.moveWindow("RGB", 0, 0)
            cv2.imshow("RGB", img) # Show image

            cv2.namedWindow("Depth")
            cv2.moveWindow("Depth", max_width+100, 0)
            cv2.imshow("Depth", zimg)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    

    def visualize_pointcloud(self, rgbimg, xyzimg, boxes, masks, zt, ze, xyzimgname, writedir):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        height, width = rgbimg.shape[:2]

        max_height = 700
        max_width = 700

        stop_program = False

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            broc_mask_binary = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],maskstransposed.shape[2]),dtype=np.uint8)

            all_masks = np.zeros((height,width,3),dtype=np.uint8)
            red_mask = np.zeros((height,width),dtype=np.uint8)
            green_mask = np.zeros((height,width),dtype=np.uint8)

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks

                bbox = boxes[i].astype(np.uint16)
                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                
                xyz_mask = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                xyz_mask = np.multiply(xyz_mask,masksel) 
                # xyz_mask[xyz_mask[:,:,2] < 500] = np.nan

                # z_top = zt[i]
                # z_edge = ze[i]
                # x_mask = xyz_mask[:,:,0]
                # y_mask = xyz_mask[:,:,1]
                # z_mask = xyz_mask[:,:,2]

                # z_mask = np.where(np.logical_and(z_mask >= z_top, z_mask <= z_edge),z_mask,0) # this code keeps the matrix dimensions intact for visualization
                # z_mask_binary = np.minimum(z_mask,1).astype(np.uint8)

                # x_mask_final = np.multiply(x_mask, z_mask_binary)
                # y_mask_final = np.multiply(y_mask, z_mask_binary)
                # z_mask_final = np.multiply(z_mask, z_mask_binary)

                # xyz_mask = np.zeros((xyz_mask.shape[0],xyz_mask.shape[1],3)).astype(np.float64)
                # xyz_mask[:,:,0] = x_mask_final
                # xyz_mask[:,:,1] = y_mask_final
                # xyz_mask[:,:,2] = z_mask_final

                # xyz_mask[xyz_mask[:,:,2] < 500] = 0.0

                # invert the mask for better 3D visualization
                xyz_mask = np.multiply(xyz_mask, -1)

                # initialize the final mask (boolean array)
                final_mask = np.zeros((masksel.shape[0],masksel.shape[1]),dtype=np.bool)

                # make the point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))
                # pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                o3d.visualization.draw_geometries([pcd])

                # do the outlier removal
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5)
                # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5)
                display_inlier_outlier(pcd, ind)
                try:
                    o3d.visualization.draw_geometries([cl])

                    # mark non-outlier points on xyz_clipped_img
                    flat = final_mask.flatten()
                    flat[ind] = True
                    final_mask = flat.reshape(final_mask.shape, order='C')

                    ret,broc_mask = cv2.threshold(np.multiply(final_mask.astype(np.uint8),255),254,255,cv2.THRESH_BINARY)
                    output = cv2.connectedComponentsWithStats(broc_mask)
                    num_labels = output[0] # The first cell is the number of labels
                    labels = output[1] # The second cell is the label matrix
                    stats = output[2] # The third cell is the stat matrix
                    centroids = output[3] # The fourth cell is the centroid matrix

                    region_areas = []

                    for label in range(1,num_labels):
                        region_areas.append(stats[label, cv2.CC_STAT_AREA])

                    if len(region_areas) > 1:
                        for w in range(len(np.asarray(region_areas))):
                            region_area = np.asarray(region_areas)[w]
                            if region_area < 500:
                                broc_mask = np.where(labels==(w+1), 0, broc_mask) 

                except:
                    broc_mask = np.zeros(((masksel.shape[0],masksel.shape[1]))).astype(np.uint8) 

                masksel_2d = masksel.reshape(masksel.shape[0],masksel.shape[1])
                mask_diff = np.subtract(np.multiply(masksel_2d,255), broc_mask)
                broc_binary = np.minimum(broc_mask,1).astype(np.uint8)
                broc_mask_binary[bbox[1]:bbox[3],bbox[0]:bbox[2],i] = broc_binary 
                
                red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff
                green_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = broc_mask

                # # xyz_mask = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                # broc_binary_3d = np.repeat(np.expand_dims(broc_binary, axis = 2), 3, axis=2)
                # xyz_mask = np.multiply(xyz_mask, broc_binary_3d)
                # xyz_mask[xyz_mask[:,:,2] == 0.0] = np.nan

                # pcdf = o3d.geometry.PointCloud()
                # pcdf.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))
                # pcdf = pcdf.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                # try:
                #     o3d.visualization.draw_geometries([pcdf])
                # except:
                #     print("No XYZ masks to visualize")

            all_masks[:,:,1] = green_mask
            all_masks[:,:,2] = red_mask
            img_mask = cv2.addWeighted(rgbimg,1,all_masks,0.5,0)

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

            cv2.namedWindow("RGB masks (red) and XYZ masks (green)")
            cv2.moveWindow("RGB masks (red) and XYZ masks (green)", 0, 0)
            cv2.imshow("RGB masks (red) and XYZ masks (green)", img_mask)
            k = cv2.waitKey(0)

            if k == ord('s'):
                xyzimg_name = os.path.basename(xyzimgname)
                src = xyzimgname
                dst = os.path.join(writedir,xyzimg_name)
                copyfile(src, dst) 

                basename = os.path.splitext(xyzimg_name)[0]
                np_name = basename + ".npy"

                with open(os.path.join(writedir,np_name), 'wb') as f:
                    np.save(f, broc_mask_binary)

                print("Image and Masks saved to file!")
                cv2.destroyAllWindows()
            elif k == 27: 
                cv2.destroyAllWindows()
            elif k == ord('q'):
                stop_program = True
                cv2.destroyAllWindows()
        else:
            broc_mask_binary = np.array([])

        return broc_mask_binary, stop_program


    def pointcloud_for_regression(self, rgbimg, xyzimg, boxes, masks, xyzcoords, diameter, cx, cy, xyzimgname, writedir, min_x, max_x, min_y, max_y, min_z, max_z):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        height, width = rgbimg.shape[:2]

        max_height = 700
        max_width = 700

        stop_program = False
        storemask = False

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            broc_mask_binary = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],maskstransposed.shape[2]),dtype=np.uint8)
            # broc_mask_store = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.uint8)

            all_masks = np.zeros((height,width,3),dtype=np.uint8)
            red_mask = np.zeros((height,width),dtype=np.uint8)
            green_mask = np.zeros((height,width),dtype=np.uint8)

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks

                bbox = boxes[i].astype(np.uint16)

                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                
                xyz_mask = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                xyz_mask = np.multiply(xyz_mask,masksel) 

                # invert the mask for better 3D visualization
                xyz_mask = np.multiply(xyz_mask, -1)

                # initialize the final mask (boolean array)
                final_mask = np.zeros((masksel.shape[0],masksel.shape[1]),dtype=np.bool)

                # make the point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))
                # pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                # o3d.visualization.draw_geometries([pcd])

                # do the outlier removal
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5)
                # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5)
                # display_inlier_outlier(pcd, ind)
                
                try:
                    bb = cl.get_axis_aligned_bounding_box()
                    bb.color = (1,0,0)
                    # bbpnts = np.asarray(bb.get_box_points())
                    # bbextent = np.asarray(bb.get_extent())
                    # bbcenter = np.asarray(bb.get_center())

                    # o3d.visualization.draw_geometries([cl, bb])

                    # hull, _ = cl.compute_convex_hull()

                    # center_pcd = o3d.geometry.Geometry3D.get_center(cl)
                    # min_pcd = o3d.geometry.Geometry3D.get_min_bound(cl)
                    # max_pcd = o3d.geometry.Geometry3D.get_max_bound(cl)

                    # center_pcd[2] = max_pcd[2]

                    cp = o3d.geometry.PointCloud()
                    cpc = np.asarray(np.multiply(xyzcoords, -1))
                    cp.points = o3d.utility.Vector3dVector(cpc.reshape(-1, 3))
                    cp.paint_uniform_color((0,0,0))
                    # o3d.visualization.draw_geometries([cl, bb, cp])

                    # mark non-outlier points on xyz_clipped_img
                    flat = final_mask.flatten()
                    flat[ind] = True
                    final_mask = flat.reshape(final_mask.shape, order='C')

                    ret,broc_mask = cv2.threshold(np.multiply(final_mask.astype(np.uint8),255),254,255,cv2.THRESH_BINARY)
                    output = cv2.connectedComponentsWithStats(broc_mask)
                    num_labels = output[0] # The first cell is the number of labels
                    labels = output[1] # The second cell is the label matrix
                    stats = output[2] # The third cell is the stat matrix
                    centroids = output[3] # The fourth cell is the centroid matrix

                    region_areas = []

                    for label in range(1,num_labels):
                        region_areas.append(stats[label, cv2.CC_STAT_AREA])

                    if len(region_areas) > 1:
                        for w in range(len(np.asarray(region_areas))):
                            region_area = np.asarray(region_areas)[w]
                            if region_area < 500:
                                broc_mask = np.where(labels==(w+1), 0, broc_mask) 

                except:
                    broc_mask = np.zeros(((masksel.shape[0],masksel.shape[1]))).astype(np.uint8) 

                masksel_2d = masksel.reshape(masksel.shape[0],masksel.shape[1])
                mask_diff = np.subtract(np.multiply(masksel_2d,255), broc_mask)
                broc_binary = np.minimum(broc_mask,1).astype(np.uint8)
                broc_mask_binary[bbox[1]:bbox[3],bbox[0]:bbox[2],i] = broc_binary 
                
                red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff
                green_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = broc_mask

                ax = bbox[0]
                ay = bbox[1]
                bx = bbox[2]
                by = bbox[3]

                if ax < cx < bx and ay < cy < by:
                    yolact_mask_store = masksel
                    broc_mask_store = broc_binary
                    xyzstore = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                    imgstore = rgbimg[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                    min_pcd = o3d.geometry.Geometry3D.get_min_bound(cl)
                    max_pcd = o3d.geometry.Geometry3D.get_max_bound(cl)

            all_masks[:,:,1] = green_mask
            all_masks[:,:,2] = red_mask
            img_mask = cv2.addWeighted(rgbimg,1,all_masks,0.5,0)

            z_img = xyzimg[:,:,2]
            np.clip(z_img, 500, 1000, out=z_img)
            z_img = np.interp(z_img, (z_img.min(), z_img.max()), (0, 200))
            z3 = cv2.cvtColor(z_img.astype(np.uint8),cv2.COLOR_GRAY2RGB) 
            z_mask = cv2.addWeighted(z3,1,all_masks,0.5,0)

            xyzf = np.multiply(xyzstore, np.repeat(np.expand_dims(broc_mask_store, axis=2), 3, axis=2))
            imgf = np.multiply(imgstore, np.repeat(yolact_mask_store, 3, axis=2))

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image
                z_mask = cv2.resize(z_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                # xyzf = cv2.resize(xyzf.astype(np.uint8), None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            # if xyzf.shape[0] < 224 and xyzf.shape[1] < 224:
            #     zp = np.zeros((224,224,3))
            #     diffx = int(np.divide(224 - xyzf.shape[0], 2))
            #     diffy = int(np.divide(224 - xyzf.shape[1], 2))
            #     zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf
            #     storemask = True
            # else:
            #     storemask = False

            # maxdim = 400
            # zp = np.zeros((maxdim,maxdim,3)).astype(np.uint8)
            # diffx = int(np.divide(maxdim - imgf.shape[0], 2))
            # diffy = int(np.divide(maxdim - imgf.shape[1], 2))
            # zp[diffx:diffx+imgf.shape[0], diffy:diffy+imgf.shape[1]] = imgf

            maxdim = 400
            zp = np.zeros((maxdim,maxdim,3)).astype(np.float32)
            diffx = int(np.divide(maxdim - xyzf.shape[0], 2))
            diffy = int(np.divide(maxdim - xyzf.shape[1], 2))
            zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf.astype(np.float32)

            print(zp.shape)
            print(zp.dtype)

            # maxdim = np.maximum(xyzf.shape[0], xyzf.shape[1])
            # if xyzf.shape[0] < 224 and xyzf.shape[1] < 224:
            #     zp = np.zeros((224,224,3))
            #     diffx = int(np.divide(224 - xyzf.shape[0], 2))
            #     diffy = int(np.divide(224 - xyzf.shape[1], 2))
            #     zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf
            # else:
            #     maxdim = np.maximum(xyzf.shape[0], xyzf.shape[1])
            #     zp = np.zeros((maxdim,maxdim,3))
            #     if xyzf.shape[0] < xyzf.shape[1]:
            #         diffx = int(np.divide(maxdim - xyzf.shape[0], 2))
            #         zp[diffx:diffx+xyzf.shape[0], :xyzf.shape[1]] = xyzf
            #     elif xyzf.shape[1] < xyzf.shape[0]:
            #         diffy = int(np.divide(maxdim - xyzf.shape[1], 2))
            #         zp[:xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf
            #     else:
            #         zp[:xyzf.shape[0], :xyzf.shape[1]] = xyzf

            # print(zp.shape)
            # zzp = zp[:,:,2]
            # zzp = np.interp(zzp, (np.multiply(max_pcd[2],-1), np.multiply(min_pcd[2],-1)), (75, 255))

            cv2.namedWindow("RGB masks (red) and XYZ masks (green)")
            cv2.moveWindow("RGB masks (red) and XYZ masks (green)", 0, 0)
            cv2.imshow("RGB masks (red) and XYZ masks (green)", img_mask)
            # cv2.namedWindow("ZP_RGB")
            # cv2.moveWindow("ZP_RGB", max_width+100, 0)
            # cv2.imshow("ZP_RGB", zp)
            k = cv2.waitKey(1)


            xyzimg_name = os.path.basename(xyzimgname)

            basename = os.path.splitext(xyzimg_name)[0]
            # basename = basename.replace("xyz", "rgb")
            # img_name = basename + ".jpg"
            img_name = basename + ".tiff"
            np_name = basename + ".npy"
            txt_name = basename + ".txt"

            # cv2.imwrite(os.path.join("/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/rgb_masks",img_name), zp)
            with open(os.path.join(writedir,np_name), 'wb') as f:
                np.save(f, zp)

                min_x_mask = np.min(zp[:,:,0])
                max_x_mask = np.max(zp[:,:,0])
                min_y_mask = np.min(zp[:,:,1])
                max_y_mask = np.max(zp[:,:,1])
                min_z_mask = np.min(zp[:,:,2])
                max_z_mask = np.max(zp[:,:,2])

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

            # imsave(os.path.join("/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/xyz_masks",img_name),zp)
            txtfile = open(os.path.join(writedir,txt_name),"w")
            txtfile.write("{0:.1f}".format(xyz_coordinates[2]))
            # txtfile.write("{0:.1f}, {1:.1f}, {2:.1f}, {3:.1f}".format(xyz_coordinates[0], xyz_coordinates[1], xyz_coordinates[2], diameter))  
            txtfile.close() 

            if storemask:
                with open(os.path.join(writedir,np_name), 'wb') as f:
                    print(zp.shape)
                    np.save(f, zp)

                txtfile = open(os.path.join(writedir,txt_name),"w")
                txtfile.write("{0:.1f}, {1:.1f}, {2:.1f}, {3:.1f}".format(xyz_coordinates[0], xyz_coordinates[1], xyz_coordinates[2], diameter)) 
                txtfile.close() 

                print("Image and labels saved to file!")
                cv2.destroyAllWindows()

            # if k == ord('s'):
            #     xyzimg_name = os.path.basename(xyzimgname)
            #     src = xyzimgname
            #     dst = os.path.join(writedir,xyzimg_name)
            #     copyfile(src, dst) 

            #     basename = os.path.splitext(xyzimg_name)[0]
            #     np_name = basename + ".npy"

            #     with open(os.path.join(writedir,np_name), 'wb') as f:
            #         np.save(f, broc_mask_binary)

            #     print("Image and Masks saved to file!")
            #     cv2.destroyAllWindows()
            # elif k == 27: 
            #     cv2.destroyAllWindows()
            # elif k == ord('q'):
            #     stop_program = True
            #     cv2.destroyAllWindows()
        else:
            broc_mask_binary = np.array([])

        return broc_mask_binary, stop_program, min_x, max_x, min_y, max_y, min_z, max_z


    def store_rle_bitmask(self, rgbimg, xyzimg, boxes, masks):  
        max_height = 700
        max_width = 700

        x = np.expand_dims(xyzimg[:,:,0], axis=2)
        y = np.expand_dims(xyzimg[:,:,1], axis=2)
        z = np.expand_dims(xyzimg[:,:,2], axis=2)

        height, width = rgbimg.shape[:2]
        red = np.expand_dims(rgbimg[:,:,0], axis=2)
        green = np.expand_dims(rgbimg[:,:,1], axis=2)
        blue = np.expand_dims(rgbimg[:,:,2], axis=2)

        writedata = {}
        writedata['info'] = {"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-05-12 17:27:18.697772"}
        writedata['licenses'] = []
        writedata['licenses'].append({"url": None, "id": 0, "name": None})
        writedata['images'] = []
        writedata['type'] = "instances"
        writedata['annotations'] = []
        writedata['categories'] = []
        writedata['categories'].append({"supercategory": "broccoli", "id": 1, "name": "broccoli"})
        annotation_id = 0

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks

                bbox = boxes[i].astype(np.uint16)

                x_mask = np.multiply(x,masksel)
                y_mask = np.multiply(y,masksel)
                z_mask = np.multiply(z,masksel)

                z_top = 630
                z_edge = 660

                x_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),x_mask,0)
                y_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),y_mask,0)
                z_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),z_mask,0) # this code keeps the matrix dimensions intact for visualization
                
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) 

                mask_vis = np.multiply(z_mask_final_binary,255)
                if max_height < height or max_width < width: # only shrink if img is bigger than required
                    scaling_factor = max_height / float(height) # get scaling factor
                    if max_width/float(width) < scaling_factor:
                        scaling_factor = max_width / float(width)
                    mask_vis = cv2.resize(mask_vis, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

                cv2.namedWindow("mask")
                cv2.moveWindow("mask", 0, 0)
                cv2.imshow("mask", mask_vis) # Show image
                cv2.waitKey(0)

                bitmask = pycocotools.mask.encode(np.asarray(z_mask_final_binary, order="F"))
                bitmask[0]['counts'] = bitmask[0]['counts'].decode("ascii")

                bbxs,bbys,bbxxs,bbyys = bbox
                bbws = bbxxs - bbxs
                bbhs = bbyys - bbys
                bb_area = int(bbws*bbhs)
                bbpoints = np.asarray((bbxs,bbys,bbws,bbhs)).flatten().tolist()

                writedata['annotations'].append({
                    'id': i,
                    'image_id': 0,
                    'category_id': 1,
                    'segmentation': bitmask[0],
                    'area': bb_area,
                    'bbox': bbpoints,
                    'iscrowd': 0
                })

        with open("/home/pieterdeeplearn/harvestcnn/datasets/train/annotations_bitmask.json", 'w') as outfile:
            json.dump(writedata, outfile)

        return 


    def prepare_donut(self):
        filenames = []

        modaldir = "/home/pieterdeeplearn/harvestcnn/datasets/donut"

        writedir = "/home/pieterdeeplearn/harvestcnn/datasets/donut/bitmasks"

        subfolders = ["train","val"]

        max_height = 750
        max_width = 750

        breakout = False

        imgcount = 0

        for i in range(len(subfolders)):
            filelist = os.listdir(os.path.join(modaldir,subfolders[i]))
            
            writedata = {}
            writedata['info'] = {"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-05-12 17:27:18.697772"}
            writedata['licenses'] = []
            writedata['licenses'].append({"url": None, "id": 0, "name": None})
            writedata['images'] = []
            writedata['type'] = "instances"
            writedata['annotations'] = []
            writedata['categories'] = []
            writedata['categories'].append({"supercategory": "donut", "id": 1, "name": "donut"})
            annotation_id = 0

            for j in range (len(filelist)):
                fname = filelist[j]
                if fname.endswith(".jpg"):
                    imgname = fname
                    print(imgcount)
                    print(imgname)
                    
                    img = cv2.imread(os.path.join(modaldir,subfolders[i],imgname))
                    height, width = img.shape[:2]
                    
                    json_name = imgname.split(".jpg")
                    jn = json_name[0]+'.json'
                    
                    cv2.imwrite(os.path.join(writedir,subfolders[i],"JPEGImages",imgname),img)
                    
                    bbox = []
                    circle = []
                    
                    writedata['images'].append({
                                    'license': 0,
                                    'url': None,
                                    'file_name': str("JPEGImages/" + imgname),
                                    'height': height,
                                    'width': width,
                                    'date_captured': None,
                                    'id': imgcount
                                })                       

                    num_instances = len(bbox)
                    visible_instances = []

                    for w in range(num_instances):
                        visible_instances.append([])
                    
                    # create visible mask regions   
                    with open(os.path.join(modaldir,subfolders[i],jn), 'r') as json_file:
                        data = json.load(json_file)
                        for p in data['shapes']:
                            if p['label'] == 'donut': 
                                donut_img = np.zeros((height,width,1),dtype=(np.uint8)) 
                                points = p['points']
                                x = [points[idx][0] for idx in range(len(points))]
                                y = [points[idx][1] for idx in range(len(points))]
                                xy = list(zip(x, y))
                                donutpoly = np.array(points).reshape(-1,1,2)
                                donut_img = cv2.fillPoly(donut_img,[donutpoly.astype(np.int32)],(255))

                                for q in data['shapes']:
                                    if q['label'] == 'hole':
                                        hole_img = np.zeros((height,width,1),dtype=(np.uint8))
                                        points = q['points']
                                        x = [points[idx][0] for idx in range(len(points))]
                                        y = [points[idx][1] for idx in range(len(points))]
                                        xy = list(zip(x, y))
                                        holepoly = np.array(points).reshape(-1,1,2)  
                                        hole_img = cv2.fillPoly(hole_img,[holepoly.astype(np.int32)],(255))

                                        intersection = cv2.bitwise_and(donut_img,hole_img)
                                        ret,thresh1 = cv2.threshold(intersection,127,255,cv2.THRESH_BINARY)

                                        output = cv2.connectedComponentsWithStats(thresh1)
                                        num_labels = output[0] # The first cell is the number of labels
                                        labels = output[1] # The second cell is the label matrix
                                        stats = output[2] # The third cell is the stat matrix
                                        centroids = output[3] # The fourth cell is the centroid matrix

                                        region_areas = []

                                        for label in range(1,num_labels):
                                            region_areas.append(stats[label, cv2.CC_STAT_AREA])

                                        if region_areas:
                                            donut_img_binary = np.minimum(donut_img,1).astype(np.uint8)
                                            hole_img_inv = cv2.bitwise_not(hole_img)
                                            cut_region = np.minimum(hole_img_inv,1).astype(np.uint8)

                                            final_mask = np.multiply(donut_img_binary, np.expand_dims(cut_region,axis=2))
                                            final_mask_vis = np.multiply(final_mask,255).astype(np.uint8)

                                            # final_mask = donut_img_binary
                                            # final_mask_vis = np.multiply(final_mask,255).astype(np.uint8)

                                            ret,thresh2 = cv2.threshold(final_mask_vis,127,255,cv2.THRESH_BINARY)
                                            contours, hier = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                            cnt = np.concatenate(contours)
                                            bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
                                            bb_area = int(bbw*bbh)
                                            bbpoints = np.asarray((bbx,bby,bbw,bbh)).flatten().tolist()

                                            bitmask = pycocotools.mask.encode(np.asarray(final_mask, order="F"))
                                            bitmask[0]['counts'] = bitmask[0]['counts'].decode("ascii")

                                            writedata['annotations'].append({
                                                'id': annotation_id,
                                                'image_id': imgcount,
                                                'category_id': 1,
                                                'segmentation': bitmask[0],
                                                'area': bb_area,
                                                'bbox': bbpoints,
                                                'iscrowd': 0
                                            })
                                    
                                            annotation_id = annotation_id+1

                                            # cv2.rectangle(final_mask_vis , (bbx, bby), (bbx+bbw, bby+bbh), (255), 1)

                                            # if max_height < height or max_width < width: # only shrink if img is bigger than required
                                            #     scaling_factor = max_height / float(height) # get scaling factor
                                            #     if max_width/float(width) < scaling_factor:
                                            #         scaling_factor = max_width / float(width)
                                            #     final_mask_vis = cv2.resize(final_mask_vis, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image
                                        
                                            # cv2.namedWindow('final_mask')
                                            # cv2.moveWindow('final_mask', 0, 0)   
                                            # cv2.imshow('final_mask', final_mask_vis)
                                            # cv2.waitKey(0)
                imgcount = imgcount+1

            with open(os.path.join(writedir,subfolders[i],"annotations.json"), 'w') as outfile:
                json.dump(writedata, outfile)


    def xyz_masks_to_json(self, modaldir, writedir):
        filenames = []

        max_height = 750
        max_width = 750

        imgcount = 0

        filelist = os.listdir(modaldir)
        
        writedata = {}
        writedata['info'] = {"description": None, "url": None, "version": None, "year": 2020, "contributor": None, "date_created": "2020-05-12 17:27:18.697772"}
        writedata['licenses'] = []
        writedata['licenses'].append({"url": None, "id": 0, "name": None})
        writedata['images'] = []
        writedata['type'] = "instances"
        writedata['annotations'] = []
        writedata['categories'] = []
        writedata['categories'].append({"supercategory": "broccoli", "id": 1, "name": "broccoli"})
        annotation_id = 0

        for j in range (len(filelist)):
            fname = filelist[j]
            if fname.endswith(".tif"):
                imgname = fname
                print(imgcount)
                print(imgname)

                img = cv2.imread(os.path.join(modaldir,imgname),-1)

                # be aware opencv2 reads an image in reversed order (so RGB->BGR and XYZ->ZYX)
                img = img[...,::-1]
                height, width = img.shape[:2]
                
                maskname = imgname.split(".tif")
                maskfile = maskname[0]+'.npy'
                
                bbox = []
                circle = []
                
                writedata['images'].append({
                                'license': 0,
                                'url': None,
                                'file_name': str("JPEGImages/" + imgname),
                                'height': height,
                                'width': width,
                                'date_captured': None,
                                'id': imgcount
                            })                       

                num_instances = len(bbox)
                visible_instances = []

                for w in range(num_instances):
                    visible_instances.append([])
                
                # create visible mask regions   
                
                masks = np.load(os.path.join(modaldir,maskfile))
                for i in range(masks.shape[-1]):
                    curmask = masks[:,:,i]

                    if np.max(curmask == 1):
                        curmask_uint8 = np.multiply(curmask,255).astype(np.uint8)

                        ret,thresh2 = cv2.threshold(curmask_uint8,254,255,cv2.THRESH_BINARY)
                        contours, hier = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = np.concatenate(contours)
                        bbx, bby, bbw, bbh = cv2.boundingRect(cnt)
                        bb_area = int(bbw*bbh)
                        bbpoints = np.asarray((bbx,bby,bbw,bbh)).flatten().tolist()

                        bitmask = pycocotools.mask.encode(np.asarray(curmask, order="F"))
                        bitmask['counts'] = bitmask['counts'].decode("ascii")
                        # bitmask[0]['counts'] = bitmask[0]['counts'].decode("ascii")

                        writedata['annotations'].append({
                            'id': annotation_id,
                            'image_id': imgcount,
                            'category_id': 1,
                            # 'segmentation': bitmask[0],
                            'segmentation': bitmask,
                            'visible_mask': bitmask, # only enable this line of code when training ORCNN on RGBXYZ data
                            'area': bb_area,
                            'bbox': bbpoints,
                            'iscrowd': 0
                        })
                
                        annotation_id = annotation_id+1

                        cv2.rectangle(curmask_uint8 , (bbx, bby), (bbx+bbw, bby+bbh), (255), 1)

                        if max_height < height or max_width < width: # only shrink if img is bigger than required
                            scaling_factor = max_height / float(height) # get scaling factor
                            if max_width/float(width) < scaling_factor:
                                scaling_factor = max_width / float(width)
                            curmask_uint8 = cv2.resize(curmask_uint8, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image
                    
                        cv2.namedWindow('final_mask')
                        cv2.moveWindow('final_mask', 0, 0)   
                        cv2.imshow('final_mask', curmask_uint8)
                        cv2.waitKey(1)

                imgcount = imgcount+1

        with open(os.path.join(writedir,"annotations.json"), 'w') as outfile:
            json.dump(writedata, outfile)


    # Python implementation of rs2_deproject_pixel_to_point
    def pixel_to_point(self, intrinsics, pixel, depth):

        x = (pixel[0] - intrinsics['ppx']) / intrinsics['fx']
        y = (pixel[1] - intrinsics['ppy']) / intrinsics['fy']

        r2 = x*x + y*y
        f = 1 + intrinsics['coeffs'][0]*r2 + intrinsics['coeffs'][1]*r2*r2 + intrinsics['coeffs'][4]*r2*r2*r2
        ux = x*f + 2*intrinsics['coeffs'][2]*x*y + intrinsics['coeffs'][3]*(r2 + 2*x*x)
        uy = y*f + 2*intrinsics['coeffs'][3]*x*y + intrinsics['coeffs'][2]*(r2 + 2*y*y)

        return([depth * ux, depth* uy, depth])



    def stop(self):
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
        cfg.OUTPUT_DIR = "weights/broccoli_amodal_visible"   
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

        model = build_model(cfg)
        process = ProcessImage(model, cfg)

        imgdir = "/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/annotation_files/polygon_annotations_rgb"
        writedir = "/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/xyz_masks"

        gtfile = "/home/pieterdeeplearn/harvestcnn/datasets/20200713_size_experiment_realsense/size_measurement_broccoli.ods"
        gt = get_data(gtfile)

        with open(os.path.join(imgdir, 'depth_intrin.json')) as json_file:
            intrinsics = json.load(json_file)

        if os.path.isdir(imgdir):
            all_images = os.listdir(imgdir)
            rgbimages = [x for x in all_images if "rgb" in x]
            depthimages = [x for x in all_images if "depth" in x]
            xyzimages = [x for x in all_images if "xyz" in x]
            rgbimages.sort()
            depthimages.sort()
            xyzimages.sort()


        stop_program = False
        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        min_z = 0
        max_z = 0

        for i in range(len(xyzimages)):
            xyzimage = xyzimages[i]
            xyzimgname = xyzimage.replace(".npy", ".png")

            rgbimgname = xyzimage.replace("xyz", "rgb")
            rgbimgname = rgbimgname.replace(".npy", ".png")
            print(rgbimgname)

            original_image = process.load_rgb_image(os.path.join(imgdir,rgbimgname))

            with open(os.path.join(imgdir, xyzimage), 'rb') as f:
                xyz_image = np.load(f)

            imgtensor = process.maketensor(original_image)
            image = process.preprocess_tensor(imgtensor)

            with torch.no_grad():
                model = model.eval()

                features = model.backbone(image.tensor)
                proposals, _ = model.proposal_generator(image, features, None)
                features_for_mask_head = [features[f] for f in model.roi_heads.in_features]
                pred_instances = model.roi_heads._forward_box(features_for_mask_head, proposals)

                assert pred_instances[0].has("pred_boxes") and pred_instances[0].has("pred_classes")
                instances,amodal_mask_logits = model.roi_heads._forward_amodal_mask(features_for_mask_head, pred_instances)
                instances,visible_mask_logits = model.roi_heads._forward_visible_mask(features_for_mask_head, pred_instances)
                processed_results = model._postprocess(instances, [imgtensor], image.image_sizes)[0]
                instances = processed_results["instances"].to("cpu")
                classes = instances.pred_classes.numpy()
                scores = instances.scores.numpy()
                bbox = instances.pred_boxes.tensor.numpy()

                # option is to convert the amodal_mask to a circle and then save it to a .json file so that it can be trained again
                amodal_masks = instances.pred_masks.numpy()
                modal_masks = instances.pred_visible_masks.numpy()

                # process.visualize(original_image, amodal_masks, modal_masks)
                z, zt, ze, masks, centers_x, centers_y, diameters = process.postprocess(xyz_image, modal_masks, max_depth_range_broc=100, max_depth_contribution=0.005)

                diametersmm = np.zeros(np.count_nonzero(zt))
                if np.count_nonzero(zt)>0:
                    for index, (z_top, z_edge, x, y, d, c) in enumerate(zip(zt, ze, centers_x, centers_y, diameters, classes)):
                        point_a = process.pixel_to_point(intrinsics, (x, y), z_edge) 
                        point_b = process.pixel_to_point(intrinsics, (x+1, y), z_edge)
                        pixel_per_mm = abs(point_a[0]-point_b[0])
                    
                        diametersmm[index] = pixel_per_mm * d

                real_diameter = 0
                plant_id = int(rgbimgname.split("_")[2].split("plant")[1])
                for k in range(1, len(gt['Sheet1'])):
                    diameter_data = gt['Sheet1'][k]
                    if int(diameter_data[0]) == plant_id:
                        real_diameter = diameter_data[1]

                idx = (np.abs(diametersmm - real_diameter)).argmin()
                diameter = diametersmm[idx]
                xyz_coordinates = xyz_image[centers_y[idx],centers_x[idx]]
                cx = centers_x[idx]
                cy = centers_y[idx]

                # process.visualize_masks(original_image, z, bbox, modal_masks, zt, ze, centers_x, centers_y, diameters, diametersmm, str(real_diameter))
                # final_masks, stop_program = process.visualize_pointcloud(original_image, xyz_image, bbox, modal_masks, zt, ze, os.path.join(imgdir,xyzimgname), writedir)
                final_masks, stop_program, min_x, max_x, min_y, max_y, min_z, max_z = process.pointcloud_for_regression(original_image, xyz_image, bbox, modal_masks, xyz_coordinates, diameter, cx, cy, os.path.join(imgdir,xyzimgname), writedir, min_x, max_x, min_y, max_y, min_z, max_z)

                print("min_x: " + str(min_x))
                print("max_x: " + str(max_x))
                print("min_y: " + str(min_y))
                print("max_y: " + str(max_y))
                print("min_z: " + str(min_z))
                print("max_z: " + str(max_z))

                if stop_program:
                    break