# import the libraries
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch
import numpy as np
import os
import cv2
import random

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

import open3d as o3d
import skimage.transform as transform


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


if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    if torch.cuda.is_available():
        # this might be needed for a proper definition of the test classes
        # register_coco_instances("broccoli_amodal_test", {}, "datasets/broccoli_amodal/test/annotations.json", "datasets/broccoli_amodal/test")
        # broccoli_amodal_test_metadata = MetadataCatalog.get("broccoli_amodal_test")
        # dataset_dicts_test = DatasetCatalog.get("broccoli_amodal_test")

        # configure the inference procedure
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TEST = ("broccoli_amodal_test",)
        cfg.NUM_GPUS = 2
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)
        # cfg.OUTPUT_DIR = "weights/broccoli_amodal_visible"
        cfg.OUTPUT_DIR = "weights/broccoli_amodal_temp6" # this is a trained model with RGB-RGB visible_mask_head    
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

        # read rgb and xyz image
        # imgdir = '/media/pieterdeeplearn/easystore/BroccoliData/20190625'
        # rgbimgname = '20190625_081422015_RGB_4.tif'
        # xyzimgname = '20190625_081422015_Depth_4.tif'

        imgdir = '/home/pieterdeeplearn/harvestcnn/datasets'
        rgbimgname = 'RealSense_20191101_183418_553_1572629658419.59_782_0_RGB.jpg'
        xyzimgname = 'RealSense_20191101_183418_671_1572629658411.75_798_0_Depth.tiff'    
       
        model = build_model(cfg)
        process = ProcessImage(model, cfg)
        original_image = process.load_rgb_image(os.path.join(imgdir,rgbimgname))
        imgtensor = process.maketensor(original_image)
        image = process.preprocess_tensor(imgtensor)

        temp = image.tensor.permute(2, 3, 1, 0)
        test1 = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]))
        test2 = test1.to("cpu").numpy()[...,::-1]

        r_tensor = test2[:,:,2]
        g_tensor = test2[:,:,1]
        b_tensor = test2[:,:,0]

        # this is the R (RGB) or the X (XYZ) channel
        plt.imshow(r_tensor)
        print(np.min(r_tensor))
        print(np.max(r_tensor))
        plt.show() 

        # this is the G (RGB) or the Y (XYZ) channel
        plt.imshow(g_tensor)
        print(np.min(g_tensor))
        print(np.max(g_tensor))
        plt.show() 

        # this is the B (RGB) or the Z (XYZ) channel
        plt.imshow(b_tensor)
        print(np.min(b_tensor))
        print(np.max(b_tensor))
        plt.show() 
        print("stop")

        with torch.no_grad():
            # insert this line if you want to do testing/inference, for training please comment out
            model = model.eval()

            features = model.backbone(image.tensor)

            # feature_level = features['p2']
            # for i in range(feature_level.shape[1]):
            #     curfeature_level = feature_level[0, i, :, :].to("cpu")
            #     curfeature_level = curfeature_level.numpy()
            #     plt.imshow(curfeature_level)
            #     plt.title('Feature-layer: ' + str(i))
            #     plt.show()

            # https://detectron2.readthedocs.io/tutorials/models.html#partially-execute-a-model
            # proposals, _ = model.proposal_generator(image, features, None)
            # features_for_mask_head = [features[f] for f in model.roi_heads.in_features]
            # instances = model.roi_heads._forward_box(features_for_mask_head, proposals)
            # mask_features = model.roi_heads.mask_pooler(features_for_mask_head, [x.pred_boxes for x in instances])
            # assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
            # instances = model.roi_heads.forward_with_given_boxes(features, instances)

            proposals, _ = model.proposal_generator(image, features, None)
            features_for_mask_head = [features[f] for f in model.roi_heads.in_features]
            pred_instances = model.roi_heads._forward_box(features_for_mask_head, proposals)

            assert pred_instances[0].has("pred_boxes") and pred_instances[0].has("pred_classes")
            instances,amodal_mask_logits = model.roi_heads._forward_amodal_mask(features_for_mask_head, pred_instances)
            # instances,visible_mask_logits = model.roi_heads._forward_visible_mask(features_for_mask_head, pred_instances)

            rgb_features1 = features
            rgb_features2 = model.backbone(image.tensor)
            
            features_p2 = rgb_features1['p2'].add(rgb_features2['p2'])
            features_p3 = rgb_features1['p3'].add(rgb_features2['p3'])
            features_p4 = rgb_features1['p4'].add(rgb_features2['p4'])
            features_p5 = rgb_features1['p5'].add(rgb_features2['p5'])
            features_p6 = rgb_features1['p6'].add(rgb_features2['p6'])

            # for sanity check: do a subtraction instead of an addition:
            # features_p2 = rgb_features1['p2'].sub(rgb_features2['p2'])
            # features_p3 = rgb_features1['p3'].sub(rgb_features2['p3'])
            # features_p4 = rgb_features1['p4'].sub(rgb_features2['p4'])
            # features_p5 = rgb_features1['p5'].sub(rgb_features2['p5'])
            # features_p6 = rgb_features1['p6'].sub(rgb_features2['p6'])

            vm_features = {'p2': features_p2, 'p3': features_p3, 'p4': features_p4, 'p5': features_p5, 'p6': features_p6}
            features_for_visible_mask_head = [vm_features[f] for f in model.roi_heads.in_features]          
            instances,visible_mask_logits = model.roi_heads._forward_visible_mask(features_for_visible_mask_head, pred_instances)

            processed_results = model._postprocess(instances, [imgtensor], image.image_sizes)[0]

            # visualization procedure
            instances = processed_results["instances"].to("cpu")
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            bbox = instances.pred_boxes.tensor.numpy()
            amodal_masks = instances.pred_masks.numpy()
            modal_masks = instances.pred_visible_masks.numpy()
            # process.visualize(original_image , amodal_masks, modal_masks)


        # load the original xyz data (in float32):
        # this is the preferred method (if it can be properly trained)
        # maybe the weights should be randomly initialized
        original_image = process.load_xyz_image(os.path.join(imgdir,xyzimgname))

        # convert the xyz data to uint8 (not preferred as the x and y channels will shift)
        # original_image = original_image.astype(np.uint8)

        # scale the xyz data properly before conversion to uint8
        # height, width = original_image.shape[:2]
        # x = np.expand_dims(original_image[:,:,0], axis=2)
        # y = np.expand_dims(original_image[:,:,1], axis=2)
        # z = np.expand_dims(original_image[:,:,2], axis=2)

        # np.clip(x, -500, 500, out=x)
        # x = np.interp(x, (x.min(), x.max()), (0, 255))
        # x = x.astype(np.uint8)

        # np.clip(y, -500, 500, out=y)
        # y = np.interp(y, (y.min(), y.max()), (0, 255))
        # y = y.astype(np.uint8)

        # np.clip(z, 400, 1000, out=z)
        # z = np.interp(z, (z.min(), z.max()), (0, 255))
        # z = z.astype(np.uint8)

        # original_image = np.zeros((height, width, 3)).astype(np.uint8)
        # original_image[:,:,0] = x.reshape((x.shape[0],x.shape[1]))
        # original_image[:,:,1] = y.reshape((y.shape[0],y.shape[1]))
        # original_image[:,:,2] = z.reshape((z.shape[0],z.shape[1]))

        # be carefull the realsense is only a depth image (no xyz!!!)
        imgtensor = process.maketensor(original_image)
        image = process.preprocess_tensor(imgtensor)
        
        temp = image.tensor.permute(2, 3, 1, 0)
        test1 = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]))
        test2 = test1.to("cpu").numpy()[...,::-1]

        x_tensor = test2[:,:,2]
        y_tensor = test2[:,:,1]
        z_tensor = test2[:,:,0]

        # this is the R (RGB) or the X (XYZ) channel
        # plt.imshow(x_tensor)
        # print(np.min(x_tensor))
        # print(np.max(x_tensor))
        # plt.show() 

        # this is the G (RGB) or the Y (XYZ) channel
        # plt.imshow(y_tensor)
        # print(np.min(y_tensor))
        # print(np.max(y_tensor))
        # plt.show() 

        # this is the B (RGB) or the Z (XYZ) channel
        # plt.imshow(z_tensor)
        # print(np.min(z_tensor))
        # print(np.max(z_tensor))
        # plt.show() 
        print("stop")

        masks = modal_masks

        if masks.any():
            maskstransposed = masks.transpose(1,2,0)

            for i in range (maskstransposed.shape[-1]):
                masksel = maskstransposed[:,:,i] # select the individual masks
                masksel = transform.resize(masksel, (x_tensor.shape[0], x_tensor.shape[1])).astype(np.float32)

                x_mask = np.multiply(x_tensor,masksel)
                y_mask = np.multiply(y_tensor,masksel)
                z_mask = np.multiply(z_tensor,masksel)

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
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.1)
                # display_inlier_outlier(pcd, ind)

                aabb = cl.get_axis_aligned_bounding_box()
                o3d.visualization.draw_geometries([cl, aabb])