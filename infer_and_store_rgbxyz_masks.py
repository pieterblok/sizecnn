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


    def visualize_masks(self, img, z, boxes, masks, zt, ze, cXs, cYs, diameters, pixelpermm, brocids_harvest, brocids_toosmall, brocids_toobig):       
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

                if i in brocids_harvest:
                    mask_diff = np.subtract(masksel,z_mask_final_binary)
                    mask_diff = mask_diff.reshape(mask_diff.shape[0],mask_diff.shape[1])
                    red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff

                    z_mask_final_binary = z_mask_final_binary.reshape(z_mask_final_binary.shape[0],z_mask_final_binary.shape[1])
                    green_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_mask_final_binary
                elif i in brocids_toosmall:
                    masksel = masksel.reshape(masksel.shape[0],masksel.shape[1])
                    red_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = masksel
                elif i in brocids_toobig:
                    masksel = masksel.reshape(masksel.shape[0],masksel.shape[1])
                    blue_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = masksel

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
                if k in brocids_harvest:
                    cv2.circle(img_mask, (cXs[k], cYs[k]), 7, (255, 255, 255), -1)
                    cv2.circle(img_mask, (cXs[k], cYs[k]), int(diameters[k]/2),(255,0,0),3)
                    cv2.putText(img_mask, "Z-top: {:.1f} mm".format(zt[k]),(cXs[k]+10,cYs[k]-15),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
                    cv2.putText(img_mask, "Z-edge: {:.1f} mm".format(ze[k]),(cXs[k]+10,cYs[k]+20),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
                    cv2.putText(img_mask, "Diameter: {:.1f} mm".format(np.multiply(diameters[k],pixelpermm)),(cXs[k]+10,cYs[k]+55),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))

                    cv2.circle(zimg_mask, (cXs[k], cYs[k]), 7, (0, 0, 0), -1)
                    cv2.circle(zimg_mask, (cXs[k], cYs[k]), int(diameters[k]/2), (255,0,0), 3)
                    cv2.putText(zimg_mask, "Z-top: {:.1f} mm".format(zt[k]),(cXs[k]+10,cYs[k]-15),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                    cv2.putText(zimg_mask, "Z-edge: {:.1f} mm".format(ze[k]),(cXs[k]+10,cYs[k]+20),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                    cv2.putText(zimg_mask, "Diameter: {:.1f} mm".format(np.multiply(diameters[k],pixelpermm)),(cXs[k]+10,cYs[k]+55),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
                else:
                    cv2.circle(img_mask, (cXs[k], cYs[k]), 7, (255, 255, 255), -1)
                    cv2.putText(img_mask, "Diameter: {:.1f} mm".format(np.multiply(diameters[k],pixelpermm)),(cXs[k]+10,cYs[k]+5),cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))

                    cv2.circle(zimg_mask, (cXs[k], cYs[k]), 7, (0, 0, 0), -1)
                    cv2.putText(zimg_mask, "Diameter: {:.1f} mm".format(np.multiply(diameters[k],pixelpermm)),(cXs[k]+10,cYs[k]+5),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))

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
    

    def visualize_pointcloud_v1(self, xyzimg, masks):
        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            
        x = np.expand_dims(xyzimg[:,:,0], axis=2)
        y = np.expand_dims(xyzimg[:,:,1], axis=2)
        z = np.expand_dims(xyzimg[:,:,2], axis=2)

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            ztts = np.zeros(maskstransposed.shape[-1])
            zees = np.zeros(maskstransposed.shape[-1])

            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks

                z_mask = np.multiply(z,masksel) # clip to the z-values only to the region of the mask, but keep the matrix dimensions intact for visualization
                z_mask_final = np.where(z_mask>=400,z_mask,0) # this code keeps the matrix dimensions intact for visualization
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) 

                final_mask_bool = z_mask_final_binary.astype(np.bool)
                
                xm = x[final_mask_bool].flatten()
                ym = y[final_mask_bool].flatten()
                zm = z[final_mask_bool].flatten()

                # invert the scales so that it can better visualize
                xm = xm * -1
                ym = ym * -1
                zm = zm * -1

                # calculate the distance between point-clouds:
                # https://github.com/intel-isl/Open3D/blob/master/examples/Python/Basic/pointcloud.ipynb
                # https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/pointcloud_outlier_removal.ipynb
                # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html

                pcd = o3d.geometry.PointCloud()
                xym = np.dstack((xm,ym))
                xyzm = np.dstack((xym,zm))
                xyzm = xyzm.reshape((xyzm.shape[1],xyzm.shape[2]))
                xyzm = xyzm.astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(xyzm)
                # o3d.visualization.draw_geometries([pcd])

                # remove outliers
                print("Statistical oulier removal")
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
                # display_inlier_outlier(pcd, ind)

                # o3d.visualization.draw_geometries([cl])

                pcd_center = o3d.geometry.PointCloud.get_center(pcd)
                pcd_max_bound = o3d.geometry.PointCloud.get_max_bound(pcd)
                pcd_min_bound = o3d.geometry.PointCloud.get_min_bound(pcd)

                cl_center = o3d.geometry.PointCloud.get_center(cl)
                cl_max_bound = o3d.geometry.PointCloud.get_max_bound(cl)
                cl_min_bound = o3d.geometry.PointCloud.get_min_bound(cl)
                ztts[i] = cl_max_bound[2]
                zees[i] = cl_min_bound[2]

        return ztts, zees


    def visualize_pointcloud_v2(self, xyzimg, masks, zt, ze):
        #	fit a sphere to X,Y, and Z data points
        #	returns the radius and center points of
        #	the best fit sphere
        def sphereFit(spX,spY,spZ):
            #   Assemble the A matrix
            spX = np.array(spX)
            spY = np.array(spY)
            spZ = np.array(spZ)
            A = np.zeros((len(spX),4))
            A[:,0] = spX*2
            A[:,1] = spY*2
            A[:,2] = spZ*2
            A[:,3] = 1

            #   Assemble the f matrix
            f = np.zeros((len(spX),1))
            f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
            C, residules, rank, singval = np.linalg.lstsq(A,f)

            #   solve for the radius
            t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
            radius = np.sqrt(t)

            return radius, C[0], C[1], C[2]

        def display_inlier_outlier(cloud, ind):
            inlier_cloud = cloud.select_by_index(ind)
            outlier_cloud = cloud.select_by_index(ind, invert=True)

            print("Showing outliers (red) and inliers (gray): ")
            outlier_cloud.paint_uniform_color([1, 0, 0])
            inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            
        x = np.expand_dims(xyzimg[:,:,0], axis=2)
        y = np.expand_dims(xyzimg[:,:,1], axis=2)
        z = np.expand_dims(xyzimg[:,:,2], axis=2)

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)  
            for i in range (maskstransposed.shape[-1]):
                masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8) # select the individual masks

                z_mask = np.multiply(z,masksel) # clip to the z-values only to the region of the mask, but keep the matrix dimensions intact for visualization
                z_top = zt[i]
                z_edge = ze[i]
                z_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),z_mask,0) # this code keeps the matrix dimensions intact for visualization
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) 

                final_mask_bool = z_mask_final_binary.astype(np.bool)
                
                xm = x[final_mask_bool].flatten()
                ym = y[final_mask_bool].flatten()
                zm = z[final_mask_bool].flatten()

                # invert the scales so that it can better visualize
                xm = xm * -1
                ym = ym * -1
                zm = zm * -1

                r, x0, y0, z0 = sphereFit(xm, ym, zm)
                # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                u, v = np.mgrid[0:2*np.pi:20j, 0:0.5*np.pi:10j] #plot half 3d sphere

                xx=np.cos(u)*np.sin(v)*r
                yy=np.sin(u)*np.sin(v)*r
                zz=np.cos(v)*r
                xx = xx + x0
                yy = yy + y0
                zz = zz + z0

                xlim1 = np.min(xm)
                xlim2 = np.max(xm)
                ylim1 = np.min(ym)
                ylim2 = np.max(ym)
                zlim1 = np.min(zm)
                zlim2 = np.max(zm)

                xxlim1 = np.min(xx)
                xxlim2 = np.max(xx)
                yylim1 = np.min(yy)
                yylim2 = np.max(yy)
                zzlim1 = np.min(zz)
                zzlim2 = np.max(zz)

                c = np.abs(zm)
                cm = plt.get_cmap("gist_rainbow")

                # calculate the distance between point-clouds:
                # https://github.com/intel-isl/Open3D/blob/master/examples/Python/Basic/pointcloud.ipynb
                # https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/pointcloud_outlier_removal.ipynb
                # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html

                pcd = o3d.geometry.PointCloud()
                xym = np.dstack((xm,ym))
                xyzm = np.dstack((xym,zm))
                xyzm = xyzm.reshape((xyzm.shape[1],xyzm.shape[2]))
                xyzm = xyzm.astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(xyzm)
                o3d.visualization.draw_geometries([pcd])

                # remove outliers
                print("Statistical oulier removal")
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                display_inlier_outlier(pcd, ind)

                o3d.visualization.draw_geometries([cl])

                #   3D plot of Sphere
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(xm, ym, zm, zdir='z', s=20, c=c, cmap=cm, rasterized=True)
                # ax.plot_wireframe(xx, yy, zz, color="r")
                # ax.set_xlim3d(xxlim1, xxlim2)
                # ax.set_ylim3d(yylim1, yylim2)
                # ax.set_zlim3d(zzlim1, zzlim2)
                # ax.set_xlabel('$x$ (mm)',fontsize=16)
                # ax.set_ylabel('\n$y$ (mm)',fontsize=16)
                # zlabel = ax.set_zlabel('\n$z$ (mm)',fontsize=16)
                # plt.show()

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

        # read rgb and xyz image
        imgdir = '/media/pieterdeeplearn/easystore/BroccoliData/20190625'
        rgbimgname = '20190625_081422015_RGB_4.tif'
        xyzimgname = '20190625_081422015_Depth_4.tif'
       
        model = build_model(cfg)
        process = ProcessImage(model, cfg)
        original_image = process.load_rgb_image(os.path.join(imgdir,rgbimgname))
        xyz_image = process.load_xyz_image(os.path.join(imgdir,xyzimgname))

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
            start = time.time()
            ztt, zee = process.visualize_pointcloud_v1(xyz_image, modal_masks)
            end = time.time()
            print("Pointcloud filtering: {:.3f} s".format(end-start))
            
            # post-processing on the depth-image (z-channel)
            z, zt, ze, masks, centers_x, centers_y, diameters = process.postprocess(xyz_image, modal_masks, max_depth_range_broc=100, max_depth_contribution=0.05)

            pixelpermm = 1.0
            minDiameter = 0
            maxDiameter = 1000

            if masks.any():
                diametersmm = np.divide(diameters,pixelpermm)
                brocids_harvest = np.where(np.logical_and(diametersmm >= minDiameter, diametersmm <= maxDiameter))[0].astype(np.uint8)
                brocids_toobig = np.where(diametersmm > maxDiameter)[0].astype(np.uint8)
                brocids_toosmall = np.where(diametersmm < minDiameter)[0].astype(np.uint8)
            else:
                brocids_harvest = np.array([]).astype(np.uint8)
                brocids_toobig = np.array([]).astype(np.uint8)
                brocids_toosmall = np.array([]).astype(np.uint8)

            # visualization procedure
            # process.visualize(original_image , amodal_masks, modal_masks)
            # process.visualize_masks(original_image, z, bbox, masks, zt, ze, centers_x, centers_y, diameters, pixelpermm, brocids_harvest, brocids_toosmall, brocids_toobig)
            # process.visualize_pointcloud_v2(xyz_image, masks, zt, ze)