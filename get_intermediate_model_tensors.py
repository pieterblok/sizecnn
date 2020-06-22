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
        # cfg.OUTPUT_DIR = "weights/broccoli_amodal"
        cfg.OUTPUT_DIR = "weights/broccoli_amodal_temp5" # this is a trained model with RGB-RGB visible_mask_head    
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model

        # read rgb and xyz image
        imgdir = '/media/pieterdeeplearn/easystore/BroccoliData/20190625'
        rgbimgname = '20190625_081422015_RGB_4.tif'
        xyzimgname = '20190625_081422015_Depth_4.tif'
       
        model = build_model(cfg)
        process = ProcessImage(model, cfg)
        original_image = process.load_rgb_image(os.path.join(imgdir,rgbimgname))
        # original_image = process.load_xyz_image(os.path.join(imgdir,xyzimgname))
        # original_image = original_image.astype(np.uint8)

        imgtensor = process.maketensor(original_image)
        image = process.preprocess_tensor(imgtensor)

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
            # instances = model.roi_heads._forward_invisible_mask(amodal_mask_logits, visible_mask_logits, pred_instances)

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
            instances = model.roi_heads._forward_invisible_mask(amodal_mask_logits, visible_mask_logits, pred_instances)

            processed_results = model._postprocess(instances, [imgtensor], image.image_sizes)[0]

            # visualization procedure
            instances = processed_results["instances"].to("cpu")
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            bbox = instances.pred_boxes.tensor.numpy()
            amodal_masks = instances.pred_masks.numpy()
            modal_masks = instances.pred_visible_masks.numpy()
            invisible_masks = instances.pred_invisible_masks.numpy()
            process.visualize(original_image , amodal_masks, modal_masks)