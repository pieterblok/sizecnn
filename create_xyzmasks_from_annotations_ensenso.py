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
from tifffile import imsave
import csv

class ProcessImage:
    def __init__(self):
        pass


    def scale_images(self, img1, img2, max_width, max_height, interpolation_method = cv2.INTER_LINEAR):
        height, width = img1.shape[:2]
        if max_height < height or max_width < width: # only shrink if img is bigger than required
            scaling_factor = max_height / float(height) # get scaling factor
            if max_width/float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            img1 = cv2.resize(img1, None, fx=scaling_factor, fy=scaling_factor, interpolation=interpolation_method) # resize image
            img2 = cv2.resize(img2, None, fx=scaling_factor, fy=scaling_factor, interpolation=interpolation_method) # resize image
        return img1, img2


    def visualize_masks(self, img, z, boxes, masks, amodal_masks, zt, ze, cXs, cYs, diameters, real_diameter):       
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
            amodalmaskstransposed = amodal_masks.transpose(1,2,0)

            red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
            all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8) # BGR
            
            z_img_final = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.uint8)
            mask_filtered = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.uint8)
            
            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks
                amodalmasksel = np.expand_dims(amodalmaskstransposed[:,:,i],axis=2).astype(np.uint8)

                bbox = boxes[i].astype(np.uint16)
                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                amodalmasksel = amodalmasksel[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                zsel = z[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the z-image to the boxing box to reduce memory load

                z_mask = np.multiply(zsel,masksel) # clip to the z-values only to the region of the mask, but keep the matrix dimensions intact for visualization
                z_top = zt[i]
                z_edge = ze[i]
                z_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),z_mask,0) # this code keeps the matrix dimensions intact for visualization
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) # make the depth image binary by clipping everything with value>1 to value 1 

                ## procedure to filter out the small non-connected parts
                ret,broc_mask = cv2.threshold(np.multiply(z_mask_final_binary, 255), 254, 255, cv2.THRESH_BINARY)
                
                ## deactivated this part, because the Ensenso XYZ images have many holes due to OpenGL error (too many connected components)
                # output = cv2.connectedComponentsWithStats(broc_mask)
                # num_labels = output[0] # The first cell is the number of labels
                # labels = output[1] # The second cell is the label matrix
                # stats = output[2] # The third cell is the stat matrix
                # centroids = output[3] # The fourth cell is the centroid matrix
                # region_areas = []

                # for label in range(1,num_labels):
                #     region_areas.append(stats[label, cv2.CC_STAT_AREA])

                # if len(region_areas) > 1:
                #     for w in range(len(np.asarray(region_areas))):
                #         region_area = np.asarray(region_areas)[w]
                #         if region_area < 500:
                #             broc_mask = np.where(labels==(w+1), 0, broc_mask) 
                #             # z_mask_final_binary = np.where(labels==(w+1), 0) 

                z_mask_final_binary = np.multiply(np.expand_dims(np.divide(broc_mask, 255), axis=2), z_mask_final_binary).astype(np.uint8)
                mask_filtered[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_mask_final_binary 

                z_img = z_mask_final.copy() # a z-image where the 0 values are removed
                color_range = 200 # from 0-200
                if int(z_top) != 0 and int(z_edge) != 0:
                    np.clip(z_img, z_top, z_edge, out=z_img)
                    z_img = np.interp(z_img, (z_img.min(), z_img.max()), (0, 200))
                z_img = z_img.astype(np.uint8) # make the 8-bit image of the z-values for visualization
                z_img_final[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_img

                amask_diff = np.subtract(amodalmasksel,masksel)
                amask_diff = amask_diff.reshape(amask_diff.shape[0],amask_diff.shape[1])
                red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = amask_diff

                mask_diff = np.subtract(masksel,z_mask_final_binary)
                mask_diff = mask_diff.reshape(mask_diff.shape[0],mask_diff.shape[1])
                blue_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff

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
            zimg_mask = cv2.addWeighted(z3,1,all_masks,0.6,0) # overlay the z3 image with the red full mask

            if z_negative:
                zt = np.multiply(zt, -1)
                ze = np.multiply(ze, -1)

            for k in range(cXs.size):
                cv2.circle(img_mask, (cXs[k], cYs[k]), 7, (0, 0, 0), -1)

                cv2.circle(zimg_mask, (cXs[k], cYs[k]), 7, (0, 0, 0), -1)

                bbox = boxes[k].astype(np.uint16)
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1
                font_thickness = 1

                text_str1 = "Real diameter: {} mm".format(real_diameter)
                
                text_w1, text_h1 = cv2.getTextSize(text_str1, font_face, font_scale, font_thickness)[0]

                text_pt1 = (bbox[0]-400, (cYs[k]-15))
                text_color1 = [255, 255, 255]
                text_color2 = [0, 0, 0]

                cv2.rectangle(img_mask, (text_pt1[0], text_pt1[1] + 7), (text_pt1[0] + text_w1, text_pt1[1] - text_h1 - 7), text_color1, -1)
                cv2.putText(img_mask, text_str1, text_pt1, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

                cv2.rectangle(zimg_mask, (text_pt1[0], text_pt1[1] + 7), (text_pt1[0] + text_w1, text_pt1[1] - text_h1 - 7), text_color1, -1)
                cv2.putText(zimg_mask, text_str1, text_pt1, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

        else:
            img_mask = img
            z_img_binary = np.minimum(z,1)
            zimg_mask = np.multiply(z_img_binary,200)
            mask_filtered = np.zeros((height, width, 1),dtype=np.uint8)

        return img_mask, zimg_mask, mask_filtered


    def postprocess(self, xyzimg, masks, amodal_masks, max_depth_range_broc=110, max_depth_contribution=0.005):
        masks = masks.astype(np.uint8)
        x = np.expand_dims(xyzimg[:,:,0], axis=2)
        y = np.expand_dims(xyzimg[:,:,1], axis=2)
        z = np.expand_dims(xyzimg[:,:,2], axis=2)

        height, width = xyzimg.shape[:2]
        z_negative = False
        save_image = False

        # check if the z-image has positive values or negative
        if np.min(z) < 0:
            z = np.multiply(z,-1)
            z_negative = True

        if masks.any():
            md, mh, mw = masks.shape
            masks_final = np.zeros((md,mh,mw),dtype=np.uint8)
            maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
            amodalmaskstransposed = amodal_masks.transpose(1,2,0)
            
            zts = np.zeros(maskstransposed.shape[-1])
            zes = np.zeros(maskstransposed.shape[-1])
            cXs = np.zeros(maskstransposed.shape[-1],dtype=np.uint16)
            cYs = np.zeros(maskstransposed.shape[-1],dtype=np.uint16)
            diameters = np.zeros(maskstransposed.shape[-1],dtype=np.float32)

            for i in range (maskstransposed.shape[-1]):
                masksel = maskstransposed[:,:,i]
                amodalmasksel = amodalmaskstransposed[:,:,i]

                contours, hierarchy = cv2.findContours((amodalmasksel*255).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cnt = np.concatenate(contours)
                area = cv2.contourArea(cnt)
                equi_diameter = np.sqrt(4*area/np.pi)

                M = cv2.moments(cnt)
                x = M["m10"] / M["m00"]
                y = M["m01"] / M["m00"]

                # black_img = np.zeros((mh,mw),dtype=np.uint8)
                # cntimg = cv2.drawContours(black_img, contours, -1, (255), 1)
                # cv2.circle(cntimg, (int(x), int(y)), 7, (255), -1)
                # cv2.circle(cntimg, (int(x), int(y)), int(equi_diameter/2),(255),1)
                # cv2.imshow("contours",cntimg)
                # cv2.waitKey(0)

                cXs[i] = int(x)
                cYs[i] = int(y)
                diameters[i] = equi_diameter 

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
                    # plt.hist(z_final, 10)
                    # plt.show()

                    # plt.bar(np.arange(10), norm1)
                    # plt.show()

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

        return z, zts, zes, masks_final, cXs, cYs, diameters, max_depth_contribution, save_image


    def xyzimg_for_regression(self, xyzimg, boxes, masks, amodal_masks, zt, ze, diameter, numneighbors=50, stdev=5):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        height, width = xyzimg.shape[:2]

        max_height = 700
        max_width = 700

        stop_program = False
        save_image = False
        error = False

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            amodalmaskstransposed = amodal_masks.transpose(1,2,0) 
            broc_mask_binary = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],maskstransposed.shape[2]),dtype=np.uint8)

            all_masks = np.zeros((height,width,3),dtype=np.uint8)
            red_mask = np.zeros((height,width),dtype=np.uint8)
            green_mask = np.zeros((height,width),dtype=np.uint8)

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks
                amodalmasksel = np.expand_dims(amodalmaskstransposed[:,:,i],axis=2).astype(np.float32)

                bbox = boxes[i].astype(np.uint16)
                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                amodalmasksel = amodalmasksel[bbox[1]:bbox[3],bbox[0]:bbox[2]]

                xyz_mask = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                xyz_mask = np.multiply(xyz_mask,masksel)
                z = xyz_mask[:,:,2] 
                z3 = np.repeat(np.expand_dims(z, axis=2), 3, axis=2)

                # apply histogram filtering
                xyz_mask = np.where(np.logical_and(z3 >= zt, z3 <= ze), xyz_mask, 0)

                # invert the mask for better 3D visualization
                xyz_mask = np.multiply(xyz_mask, -1)

                # initialize the final mask (boolean array)
                final_mask = np.zeros((masksel.shape[0], masksel.shape[1]), dtype=np.bool)
                xyzf = np.zeros((masksel.shape[0], masksel.shape[1], 4), dtype=np.float32)                 

                # make the point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))
                o3d.visualization.draw_geometries([pcd])

                # do the outlier removal
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=numneighbors, std_ratio=stdev)
                
                try:
                    flat = final_mask.flatten()
                    flat[ind] = True
                    final_mask = flat.reshape(final_mask.shape, order='C')

                    xyzi = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                    mask = np.repeat(np.expand_dims(final_mask, axis=2), 3, axis=2)

                    xyzf[:,:,:3] = np.multiply(xyzi, mask)
                    xyzf[:,:,3] = amodalmasksel.reshape((masksel.shape[0], masksel.shape[1]))

                    xyzf_copy = xyzf[:,:,:3].copy()
                    xyzf_copy[xyzf_copy == 0] = np.nan
                    xyzf_copy = np.multiply(xyzf_copy, -1)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyzf_copy.reshape(-1, 3))
                    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                    o3d.visualization.draw_geometries([pcd])

                    cv2.namedWindow("XYZ final")
                    cv2.moveWindow("XYZ final", 0, 0)
                    cv2.imshow("XYZ final", xyzf[:,:,:3].astype(np.uint8))

                    cv2.namedWindow("Amodal mask")
                    cv2.moveWindow("Amodal mask", masksel.shape[0] + 300, 0)
                    cv2.imshow("Amodal mask", np.multiply(xyzf[:,:,3], 255).astype(np.uint8))
                    k = cv2.waitKey(0)

                    if k == ord('r'):
                        save_image = False
                        cv2.destroyAllWindows()
                    elif k == 27: 
                        save_image = True
                        cv2.destroyAllWindows()
                    elif k == ord('q'):
                        stop_program = True
                        cv2.destroyAllWindows()
                    
                except:
                    error = True
                    break

        else:
            xyzf = np.array([])
            error = True

        return xyzf, save_image, stop_program, error, numneighbors, stdev


    def xyzimg_for_regression_ensenso(self, xyzimg, boxes, masks, amodal_masks, zt, ze, diameter, numneighbors=50, stdev=5):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        height, width = xyzimg.shape[:2]

        max_height = 700
        max_width = 700

        stop_program = False
        save_image = False
        error = False

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            amodalmaskstransposed = amodal_masks.transpose(1,2,0) 
            broc_mask_binary = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],maskstransposed.shape[2]),dtype=np.uint8)

            all_masks = np.zeros((height,width,3),dtype=np.uint8)
            red_mask = np.zeros((height,width),dtype=np.uint8)
            green_mask = np.zeros((height,width),dtype=np.uint8)

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks
                amodalmasksel = np.expand_dims(amodalmaskstransposed[:,:,i],axis=2).astype(np.float32)

                bbox = boxes[i].astype(np.uint16)
                masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]] # crop the mask to the boxing box to reduce memory load
                amodalmasksel = amodalmasksel[bbox[1]:bbox[3],bbox[0]:bbox[2]]

                xyz_mask = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                xyz_mask = np.multiply(xyz_mask,masksel)
                z = xyz_mask[:,:,2] 
                z3 = np.repeat(np.expand_dims(z, axis=2), 3, axis=2)

                # apply histogram filtering
                xyz_mask = np.where(np.logical_and(z3 >= zt, z3 <= ze), xyz_mask, 0)

                # invert the mask for better 3D visualization
                xyz_mask_copy = xyz_mask.copy()
                xyz_mask = np.multiply(xyz_mask, -1)

                # initialize the final mask (boolean array)
                final_mask = np.zeros((masksel.shape[0], masksel.shape[1]), dtype=np.bool)
                xyzf = np.zeros((masksel.shape[0], masksel.shape[1], 4), dtype=np.float32)                 

                # do the outlier removal
                if numneighbors != 50 or stdev != 5:
                    # make the point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))
                    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=numneighbors, std_ratio=stdev)
                
                    try:
                        flat = final_mask.flatten()
                        flat[ind] = True
                        final_mask = flat.reshape(final_mask.shape, order='C')

                        xyzi = xyzimg[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
                        mask = np.repeat(np.expand_dims(final_mask, axis=2), 3, axis=2)

                        xyzf[:,:,:3] = np.multiply(xyzi, mask)
                        xyzf[:,:,3] = amodalmasksel.reshape((masksel.shape[0], masksel.shape[1]))

                        xyzf_copy = xyzf[:,:,:3].copy()
                        xyzf_copy[xyzf_copy == 0] = np.nan
                        xyzf_copy = np.multiply(xyzf_copy, -1)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(xyzf_copy.reshape(-1, 3))
                        pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                        o3d.visualization.draw_geometries([pcd])

                        cv2.namedWindow("XYZ final")
                        cv2.moveWindow("XYZ final", 0, 0)
                        cv2.imshow("XYZ final", xyzf[:,:,:3].astype(np.uint8))

                        cv2.namedWindow("Amodal mask")
                        cv2.moveWindow("Amodal mask", masksel.shape[0] + 300, 0)
                        cv2.imshow("Amodal mask", np.multiply(xyzf[:,:,3], 255).astype(np.uint8))
                        k = cv2.waitKey(0)

                        if k == ord('r'):
                            save_image = False
                            cv2.destroyAllWindows()
                        elif k == 27: 
                            save_image = True
                            cv2.destroyAllWindows()
                        elif k == ord('q'):
                            stop_program = True
                            cv2.destroyAllWindows()
                        
                    except:
                        error = True
                        break
                else:
                    xyzf[:,:,:3] = xyz_mask_copy
                    xyzf[:,:,3] = amodalmasksel.reshape((masksel.shape[0], masksel.shape[1]))
                    xyzf_copy = xyzf[:,:,:3].copy()
                    xyzf_copy[xyzf_copy == (0,0,0)] = np.nan
                    xyzf_copy = np.multiply(xyzf_copy, -1)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyzf_copy.reshape(-1, 3))
                    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                    o3d.visualization.draw_geometries([pcd])

                    cv2.namedWindow("XYZ final")
                    cv2.moveWindow("XYZ final", 0, 0)
                    cv2.imshow("XYZ final", xyzf[:,:,:3].astype(np.uint8))

                    cv2.namedWindow("Amodal mask")
                    cv2.moveWindow("Amodal mask", masksel.shape[0] + 300, 0)
                    cv2.imshow("Amodal mask", np.multiply(xyzf[:,:,3], 255).astype(np.uint8))
                    k = cv2.waitKey(0)

                    if k == ord('r'):
                        save_image = False
                        cv2.destroyAllWindows()
                    elif k == 27: 
                        save_image = True
                        cv2.destroyAllWindows()
                    elif k == ord('q'):
                        stop_program = True
                        cv2.destroyAllWindows()


        else:
            xyzf = np.array([])
            error = True

        return xyzf, save_image, stop_program, error, numneighbors, stdev


    def zeropadding(self, xyzf, dimension=400):
        zp = np.zeros((dimension,dimension,xyzf.shape[-1])).astype(np.float32)
        diffx = int(np.divide(dimension - xyzf.shape[0], 2))
        diffy = int(np.divide(dimension - xyzf.shape[1], 2))
        zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf.astype(np.float32)
        
        return zp


    def pointnet(self, xyzimg, boxes, masks, zt, ze, diameter, numneighbors=50, stdev=5):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        height, width = xyzimg.shape[:2]

        max_height = 700
        max_width = 700

        stop_program = False
        save_image = False
        error = False

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
                z = xyz_mask[:,:,2] 
                z3 = np.repeat(np.expand_dims(z, axis=2), 3, axis=2)

                # apply histogram filtering
                xyz_mask = np.where(np.logical_and(z3 >= zt, z3 <= ze), xyz_mask, 0)

                # invert the mask for better 3D visualization
                xyz_mask = np.multiply(xyz_mask, -1)

                # initialize the final mask (boolean array)
                final_mask = np.zeros((masksel.shape[0], masksel.shape[1]), dtype=np.bool)                

                # make the point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_mask.reshape(-1, 3))

                # do the outlier removal
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=numneighbors, std_ratio=stdev)
                
                try:
                    bb = cl.get_axis_aligned_bounding_box()
                    o3d.visualization.draw_geometries([cl, bb])

                    empty = np.zeros((max_height, max_width, 1)).astype(np.uint8)
                    cv2.imshow("XYZ final", empty)
                    k = cv2.waitKey(0)

                    xyzf[xyzf == 0] = np.nan
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyzf.reshape(-1, 3))
                    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
                    o3d.visualization.draw_geometries([pcd])

                    if k == ord('r'):
                        save_image = False
                        cv2.destroyAllWindows()
                    elif k == 27: 
                        save_image = True
                        cv2.destroyAllWindows()
                    elif k == ord('q'):
                        stop_program = True
                        cv2.destroyAllWindows()
                    
                except:
                    error = True
                    break

        else:
            cl = np.array([])

        return cl, save_image, stop_program, error, numneighbors, stdev


    def stop(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    xyzimgdir = "/home/pieterdeeplearn/harvestcnn/datasets/20181012_size_experiment_ensenso/RGB_XYZ_images"
    amodaldir = "/home/pieterdeeplearn/harvestcnn/datasets/20181012_size_experiment_ensenso/images_and_annotations_for_diameter_estimation/circle_annotations/test"
    modaldir = "/home/pieterdeeplearn/harvestcnn/datasets/20181012_size_experiment_ensenso/images_and_annotations_for_diameter_estimation/mask_annotations/test"

    rootdir = "/home/pieterdeeplearn/harvestcnn/datasets/20181012_size_experiment_ensenso/xyz_masks"
    writedir = os.path.join(rootdir, "test")

    gtfile = "/home/pieterdeeplearn/harvestcnn/datasets/20181012_size_experiment_ensenso/size_measurement_broccoli.ods"
    gt = get_data(gtfile)

    redodir = "/home/pieterdeeplearn/Desktop/redodir"
    redofile = "/home/pieterdeeplearn/Desktop/redo.ods"
    redo = get_data(redofile)

    if os.path.isdir(amodaldir):
        all_files = os.listdir(amodaldir)
        amodal_images = [x for x in all_files if "rgb" in x and ".png" in x]
        amodal_annotations = [x for x in all_files if "rgb" in x and ".json" in x]
        amodal_images.sort()
        amodal_annotations.sort()

    if os.path.isdir(modaldir):
        all_files = os.listdir(modaldir)
        modal_images = [x for x in all_files if "rgb" in x and ".png" in x]
        modal_annotations = [x for x in all_files if "rgb" in x and ".json" in x]
        modal_images.sort()
        modal_annotations.sort()

    stop_program = False
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    min_z = 9999
    max_z = 0

    if not os.path.isfile(os.path.join(rootdir, 'images_that_cannot_be_processed.csv')):
        with open(os.path.join(rootdir, 'images_that_cannot_be_processed.csv'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['image_name'])
    
    start_i = 0

    for i in range(start_i, len(amodal_images)):
        print(i)
        imgname = amodal_images[i]
        img = cv2.imread(os.path.join(amodaldir, imgname))
        height, width = img.shape[:2]

        annotation = imgname.replace(".png", ".json")
        xyzimgname = imgname.replace(".png", ".tiff")
        xyzimgname = xyzimgname.replace("rgb", "xyz")

        print(imgname)

        basename, fileext = os.path.splitext(imgname)
        plant_id = int(basename.split("_")[0])

        process = ProcessImage()

        ## load the xyz image
        xyz_image = tifffile.imread(os.path.join(xyzimgdir, xyzimgname))

        ## load the modal mask (circle)
        amodal_mask = np.zeros((height, width, 1)).astype(np.uint8)
        with open(os.path.join(amodaldir, annotation), 'r') as amodal_json:
            data = json.load(amodal_json)
            for p in data['shapes']:
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

                amodal_mask = cv2.circle(amodal_mask, centerpoint, radius, (255), -1)


        ## load the modal mask (polygon)
        modal_mask = np.zeros((height, width, 1)).astype(np.uint8)
        with open(os.path.join(modaldir, annotation), 'r') as modal_json:
            data = json.load(modal_json)
            for p in data['shapes']:
                pts = []
                for k in range(len(p['points'])):
                    pts.append(p['points'][k])

                points = np.array(pts).astype(np.int32)
                modal_mask = cv2.fillPoly(modal_mask, [points], [255], lineType=cv2.LINE_AA)

        ret,thresh = cv2.threshold(amodal_mask,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        nested_array = [[x, y, (x+w), (y+h)]]
        bb = np.asarray(nested_array).astype(np.float32)

        amodal_masks = amodal_mask.transpose(2,0,1).astype(np.bool)
        modal_masks = modal_mask.transpose(2,0,1).astype(np.bool)

        mdc = 0.02
        z, ztop, zedge, masks, centers_x, centers_y, diameters, mdc, save_image_pp = process.postprocess(xyz_image, modal_masks, amodal_masks, max_depth_range_broc=90, max_depth_contribution=mdc)
        
        real_diameter = 0
        for k in range(1, len(gt['Sheet1'])):
            diameter_data = gt['Sheet1'][k]
            if int(diameter_data[0]) == plant_id:
                real_diameter = diameter_data[1]

        img_mask, zimg_mask, modal_mask_final = process.visualize_masks(img, z, bb, modal_masks, amodal_masks, ztop, zedge, centers_x, centers_y, diameters, str(real_diameter))
        img_mask, zimg_mask = process.scale_images(img_mask, zimg_mask, 700, 700)    
        
        cv2.namedWindow("RGB Mask")
        cv2.moveWindow("RGB Mask", 0, 0)
        cv2.imshow("RGB Mask", img_mask) # Show image

        cv2.namedWindow("Z Mask")
        cv2.moveWindow("Z Mask", 800, 0)
        cv2.imshow("Z Mask", zimg_mask)
        k = cv2.waitKey(0)

        if k == ord('r'):
            save_image_pp = False
            cv2.destroyAllWindows()
        elif k == 27: 
            save_image_pp = True
            cv2.destroyAllWindows()

        while not save_image_pp:
            mdc = mdc + 0.02
            z, ztop, zedge, masks, centers_x, centers_y, diameters, mdc, save_image_pp = process.postprocess(xyz_image, modal_masks, amodal_masks, max_depth_range_broc=90, max_depth_contribution=mdc)

            real_diameter = 0
            for k in range(1, len(gt['Sheet1'])):
                diameter_data = gt['Sheet1'][k]
                if int(diameter_data[0]) == plant_id:
                    real_diameter = diameter_data[1]

            img_mask, zimg_mask, modal_mask_final = process.visualize_masks(img, z, bb, modal_masks, amodal_masks, ztop, zedge, centers_x, centers_y, diameters, str(real_diameter))
            img_mask, zimg_mask = process.scale_images(img_mask, zimg_mask, 700, 700)    
            
            cv2.namedWindow("RGB Mask")
            cv2.moveWindow("RGB Mask", 0, 0)
            cv2.imshow("RGB Mask", img_mask) # Show image

            cv2.namedWindow("Z Mask")
            cv2.moveWindow("Z Mask", 800, 0)
            cv2.imshow("Z Mask", zimg_mask)
            k = cv2.waitKey(0)

            if k == ord('r'):
                save_image_pp = False
                cv2.destroyAllWindows()
            elif k == 27: 
                save_image_pp = True
                cv2.destroyAllWindows()

            if save_image_pp:
                break

        am = amodal_masks
        mm = modal_mask_final.transpose(2,0,1).astype(np.bool)

        nn = 50
        sd = 5
        xyzf, save_image, stop_program, error, nn, sd = process.xyzimg_for_regression_ensenso(xyz_image, bb, mm, am, ztop, zedge, real_diameter, nn, sd) 

        if not error and not stop_program:
            while not save_image:
                sd = sd-1
                if sd <= 3:
                    sd = sd - 0.25
                xyzf, save_image, stop_program, error, nn, sd = process.xyzimg_for_regression_ensenso(xyz_image, bb, mm, am, ztop, zedge, real_diameter, nn, sd)
                if save_image:
                    break

            if save_image:
                xyzimg_name = os.path.basename(xyzimgname)
                basename = os.path.splitext(xyzimg_name)[0]
                write_name = basename + ".tiff"

                zp = process.zeropadding(xyzf, 570)
                print(zp.shape)
                print(zp.dtype)

                tifffile.imsave(os.path.join(writedir, write_name), zp)

                txt_name = basename + ".txt"
                txtfile = open(os.path.join(writedir,txt_name),"w")
                txtfile.write("{0:.1f}".format(real_diameter))
                txtfile.close() 
                
                # extract the extreme x, y and z values for normalization
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

        else:
            print("image cannot be processed")
            with open(os.path.join(rootdir, 'images_that_cannot_be_processed.csv'), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow([imgname,])


        if stop_program:
            break              


print("min_x: " + str(min_x))
print("max_x: " + str(max_x))
print("min_y: " + str(min_y))
print("max_y: " + str(max_y))
print("min_z: " + str(min_z))
print("max_z: " + str(max_z))