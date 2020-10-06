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



    def postprocess(self, xyzimg, masks, amodal_masks, max_depth_range_broc=110, max_depth_contribution=0.005):
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
                        # plt.hist(z_mask_filtered, 10)
                        # plt.show()
                        
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

                    sel1 = bin_edges_f[np.where(norm1 >= max_depth_contribution)] # select the lower bound of the bins that contribute for at least 5% to the histogram depth
                    sel2 = bin_edges_f[np.where(norm1 >= max_depth_contribution)[0]+1] # select the lower bound of the bins that contribute for at least 5% to the histogram depth
                    edges = np.concatenate((sel1,sel2), axis=0) # concatenate all bins
                    final_bins = np.unique(edges) # get only the unique bins

                    #BE AWARE OF THE COORDINATE-SYSTEM: THE Z-PLANE CAN BE NEGATIVE OR POSITIVE: 
                    z_top = np.min(final_bins) # extract the z-top as minimum z-value (closest to the camera)
                    z_edge = np.max(final_bins) # extract the z-edge as maximum z-value (most remote from the camera)

                    # z_final_vis = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= z_top, z_mask_filtered <= z_edge))]
                    # plt.hist(z_final_vis, 10)
                    # plt.show()

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



    def zeropadding(self, xyzf, dimension=400):
        zp = np.zeros((dimension,dimension,xyzf.shape[-1])).astype(np.float32)
        diffx = int(np.divide(dimension - xyzf.shape[0], 2))
        diffy = int(np.divide(dimension - xyzf.shape[1], 2))
        zp[diffx:diffx+xyzf.shape[0], diffy:diffy+xyzf.shape[1]] = xyzf.astype(np.float32)
        
        return zp



    def stop(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    subfolders = ["train", "val", "test"]
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    min_z = 9999
    max_z = 0

    for j in range(len(subfolders)):
        subdir = subfolders[j]
        maindir = "/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/regression"
        xyzimgdir = os.path.join(maindir, subdir)
        amodaldir = os.path.join(maindir, "circle_annotations", subdir)
        modaldir = os.path.join(maindir, "mask_annotations", subdir)

        rootdir = "/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/xyz_masks"
        writedir = os.path.join(rootdir, subdir)

        gtfile = "/home/pieterdeeplearn/harvestcnn/datasets/train_val_test_files/size_measurement_broccoli.ods"
        gt = get_data(gtfile)

        if os.path.isdir(xyzimgdir):
            all_files = os.listdir(xyzimgdir)
            xyz_images = [x for x in all_files if "xyz" in x and ".tiff" in x]
            xyz_images.sort()

        if os.path.isdir(amodaldir):
            all_files = os.listdir(amodaldir)
            amodal_annotations = [x for x in all_files if "rgb" in x and ".json" in x]
            amodal_annotations.sort()

        if os.path.isdir(modaldir):
            all_files = os.listdir(modaldir)
            modal_annotations = [x for x in all_files if "rgb" in x and ".json" in x]
            modal_annotations.sort()

        stop_program = False
        
        start_i = 0

        for i in range(start_i, len(xyz_images)):
            print(i)

            ## load the xyz image
            xyzimgname = xyz_images[i]
            xyz_image = tifffile.imread(os.path.join(xyzimgdir, xyzimgname))
            height, width = xyz_image.shape[:2]
            print(xyzimgname)


            ## find the annotation name
            annotation = xyzimgname.replace(".tiff", ".json")
            annotation = annotation.replace("xyz", "rgb")
            basename, fileext = os.path.splitext(xyzimgname)


            ## find the plant id
            if "plant" in basename:
                plant_id = int(basename.split("_")[2].split("plant")[1])
            else:
                plant_id = int(basename.split("_")[0])

            process = ProcessImage()


            ## get the ground truth diameter
            real_diameter = 0
            for k in range(1, len(gt['size_measurement'])):
                diameter_data = gt['size_measurement'][k]
                if int(diameter_data[0]) == plant_id:
                    real_diameter = diameter_data[1]


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

            amodal_mask_binary = np.minimum(amodal_mask, 1)


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

            modal_mask_binary = np.minimum(modal_mask, 1)
            modal_mask_binary = np.repeat(modal_mask_binary, 3, axis=2)


            ## get the bounding box from the amodal mask
            ret,thresh = cv2.threshold(amodal_mask,127,255,0)
            contours,hierarchy = cv2.findContours(thresh, 1, 2)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            nested_array = [[x, y, (x+w), (y+h)]]
            bbox = np.asarray(nested_array)


            # make the final xyz mask
            xyz_img_clip = xyz_image[int(bbox[0][1]):int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2]),:]
            mmb_clip = modal_mask_binary[int(bbox[0][1]):int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2]),:]
            xyz_mask = np.multiply(xyz_img_clip, mmb_clip)
            print(xyz_mask.shape)


            ## visualize the unfiltered point cloud (from the xyz mask)
            # xyz_mask_copy = xyz_mask.copy()
            # xyz_mask_copy[xyz_mask_copy == 0] = np.nan
            # xyz_mask_copy = np.multiply(xyz_mask_copy, -1)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz_mask_copy.reshape(-1, 3))
            # pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
            # o3d.visualization.draw_geometries([pcd])
            

            ## make the final xyz+amodal mask
            xyzf = np.zeros((xyz_mask.shape[0], xyz_mask.shape[1], 4), dtype=np.float32) 
            xyzf[:,:,:3] = xyz_mask.astype(np.float32)
            amb_clip = amodal_mask_binary[int(bbox[0][1]):int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2]),:]
            xyzf[:,:,3] = amb_clip.reshape(xyz_mask.shape[0], xyz_mask.shape[1]).astype(np.float32)

            
            ## visualize the point cloud of the filtered broccoli head
            # amodal_masks = amodal_mask.transpose(2,0,1).astype(np.bool)
            # modal_masks = modal_mask.transpose(2,0,1).astype(np.bool)
            # z, ztop, zedge, masks, centers_x, centers_y, diameters = process.postprocess(xyz_image, modal_masks, amodal_masks, max_depth_range_broc=100, max_depth_contribution=0.04)
            # xyz_mask_copy = xyz_mask.copy()
            # xyz_mask_copy[np.logical_or(xyz_mask_copy[:,:,2] < ztop, xyz_mask_copy[:,:,2] > zedge)] = np.nan
            # xyz_mask_copy = np.multiply(xyz_mask_copy, -1)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz_mask_copy.reshape(-1, 3))
            # pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=False)
            # o3d.visualization.draw_geometries([pcd])    
            # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=5)
            # o3d.visualization.draw_geometries([cl])   
            

            ## apply zeropadding to resize the final mask to a fixed size
            zp = process.zeropadding(xyzf, 600)
            

            ## visualization
            cv2.namedWindow("XYZ Mask")
            cv2.moveWindow("XYZ Mask", 0, 0)
            cv2.imshow("XYZ Mask", zp[:,:,:3].astype(np.uint8))

            cv2.namedWindow("Amodal Mask")
            cv2.moveWindow("Amodal Mask", 700, 0)
            cv2.imshow("Amodal Mask", np.multiply(zp[:,:,3], 255).astype(np.uint8))
            k = cv2.waitKey(0)


            ## write the final xyz mask and its label
            xyzimg_name = os.path.basename(xyzimgname)
            basename = os.path.splitext(xyzimg_name)[0]
            write_name = basename + ".tiff"
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


print("min_x: " + str(min_x))
print("max_x: " + str(max_x))
print("min_y: " + str(min_y))
print("max_y: " + str(max_y))
print("min_z: " + str(min_z))
print("max_z: " + str(max_z))