import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

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
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

import open3d as o3d


def visualize(img, xyzimg, amodal_masks, boxes):
    amodal_masks = amodal_masks.astype(dtype=np.uint8)
    max_height = 750
    max_width = 750
    
    zimg = xyzimg[:,:,2].astype(np.uint8)

    if amodal_masks.any():
        amodal_maskstransposed = amodal_masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)

        blue_mask = np.zeros((amodal_maskstransposed.shape[0],amodal_maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((amodal_maskstransposed.shape[0],amodal_maskstransposed.shape[1],3),dtype=np.uint8) # BGR
 
        for i in range (amodal_maskstransposed.shape[-1]):
            amodal_mask = amodal_maskstransposed[:,:,i]
            blue_mask = cv2.add(blue_mask, amodal_mask)

        all_masks[:,:,2] = blue_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        height, width = img.shape[:2]
        img_mask = cv2.addWeighted(img,1,all_masks,0.5,0)

        z3 = cv2.cvtColor(zimg,cv2.COLOR_GRAY2RGB)  
        zimg_mask = cv2.addWeighted(z3,0.7,all_masks,0.9,0)

        if max_height < height or max_width < width: # only shrink if img is bigger than required
            scaling_factor = max_height / float(height) # get scaling factor
            if max_width/float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            zimg_mask = cv2.resize(zimg_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
         
        cv2.namedWindow("RGB")
        cv2.moveWindow("RGB", 0, 0)
        cv2.imshow("RGB", img_mask) # Show image

        cv2.namedWindow("Z-img")
        cv2.moveWindow("Z-img", max_width+100, 0)
        cv2.imshow("Z-img" , zimg_mask)
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


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    
if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'

    register_coco_instances("broccoli_xyz_train", {}, "datasets/broccoli_xyz/train/bitmasks/annotations.json", "datasets/broccoli_xyz/train/bitmasks")
    register_coco_instances("broccoli_xyz_val", {}, "datasets/broccoli_xyz/val/bitmasks/annotations.json", "datasets/broccoli_xyz/val/bitmasks")

    broccoli_xyz_train_metadata = MetadataCatalog.get("broccoli_xyz_train")
    broccoli_xyz_val_metadata = MetadataCatalog.get("broccoli_xyz_val")
    print(broccoli_xyz_train_metadata)
    print(broccoli_xyz_val_metadata)

    dataset_dicts_train = DatasetCatalog.get("broccoli_xyz_train")
    dataset_dicts_val = DatasetCatalog.get("broccoli_xyz_val")

    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("broccoli_xyz_train",)
    cfg.DATASETS.VAL = ("broccoli_xyz_val",)
    cfg.DATASETS.TEST = ("broccoli_xyz_val",)

    cfg.NUM_GPUS = 2
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # Let training initialize from model zoo

    # solver file settings extracted from: https://github.com/facebookresearch/Detectron/blob/master/configs/04_2018_gn_baselines/scratch_e2e_mask_rcnn_R-101-FPN_3x_gn.yaml
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.LR_POLICY = 'steps_with_decay'
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (0, 4000, 4500)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    #cfg.SOLVER.MAX_ITER = 270000
    #cfg.SOLVER.STEPS = (0, 210000, 250000)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)

    # https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    cfg.OUTPUT_DIR = "weights/broccoli_xyz"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    for d in range(len(dataset_dicts_val)):
        filename = dataset_dicts_val[d]["file_name"]
        print(d)
        print(filename)
        
        read_dir = "/media/pieterdeeplearn/easystore/BroccoliData"
        subdir = os.path.basename(filename).split("_")[0]
        rgb_imgname = os.path.basename(filename).replace("Depth", "RGB")
        rgbimg = cv2.imread(os.path.join(read_dir,subdir,rgb_imgname))
        
        # xyz procedure
        zyximg = cv2.imread(filename,-1)
        xyzimg = zyximg[...,::-1]
        outputs = predictor(zyximg)
        
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        bbox = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        
        visualize(rgbimg, xyzimg, masks, bbox)
        
        if masks.any():
            maskstransposed = masks.transpose(1,2,0)

            for i in range (maskstransposed.shape[-1]):
                masksel = np.repeat(np.expand_dims(maskstransposed[:,:,i], axis=2), 3, axis=2)

                zyx_mask = np.multiply(zyximg,masksel)
                x_mask = zyx_mask[:,:,2]
                y_mask = zyx_mask[:,:,1]
                z_mask = zyx_mask[:,:,0]

                # black color (no depth values) is -2 so filter everything that is positive
                x_mask = np.where(z_mask>=0,x_mask,0)
                y_mask = np.where(z_mask>=0,y_mask,0)
                z_mask = np.where(z_mask>=0,z_mask,0)

                z_mask_final_binary = np.minimum(z_mask,1).astype(np.uint8) 
                final_mask_bool = z_mask_final_binary.astype(np.bool)

                xm = x_mask[final_mask_bool].flatten()
                ym = y_mask[final_mask_bool].flatten()
                zm = z_mask[final_mask_bool].flatten()

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
                o3d.visualization.draw_geometries([pcd])

                print("Statistical oulier removal")
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    #             display_inlier_outlier(pcd, ind)

                aabb = cl.get_axis_aligned_bounding_box()
                o3d.visualization.draw_geometries([cl, aabb])