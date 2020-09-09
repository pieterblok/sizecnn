import detectron2
import torch
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
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
   
if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    # To solely train Mask R-CNN (without the ORCNN head), you first need to comment out the 
    # lines that involve the loading of the visible masks in the file detectron2/data/detection_utils.py)
    # comment out these lines: 298, 303, 307, 314, 326-328, 332
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

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)

    # https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
    cfg.OUTPUT_DIR = "weights/broccoli_xyz"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 

    trainer.resume_or_load(resume=False)

    # to get Mask R-CNN trained on broccoli_XYZ images (float32) follow the following procedure:
    # in detectron2/data/detection_utils.py change line 50 image = Image.open(f) to image = cv2.imread(file_name,-1)
    # comment out line 59-71
    trainer.train()