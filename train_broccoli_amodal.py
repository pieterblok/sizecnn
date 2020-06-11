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
  
if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    if torch.cuda.is_available():
        # register the amodal datasets 
        register_coco_instances("broccoli_amodal_train", {}, "datasets/broccoli_amodal/train/annotations.json", "datasets/broccoli_amodal/train")
        register_coco_instances("broccoli_amodal_val", {}, "datasets/broccoli_amodal/val/annotations.json", "datasets/broccoli_amodal/val")
        register_coco_instances("broccoli_amodal_test", {}, "datasets/broccoli_amodal/test/annotations.json", "datasets/broccoli_amodal/test")

        # create the metadata files 
        broccoli_amodal_train_metadata = MetadataCatalog.get("broccoli_amodal_train")
        broccoli_amodal_val_metadata = MetadataCatalog.get("broccoli_amodal_val")
        broccoli_amodal_test_metadata = MetadataCatalog.get("broccoli_amodal_test")

        # create the dataset dicts 
        dataset_dicts_train = DatasetCatalog.get("broccoli_amodal_train")
        dataset_dicts_val = DatasetCatalog.get("broccoli_amodal_val")
        dataset_dicts_test = DatasetCatalog.get("broccoli_amodal_test")

        # configure the training procedure, extracted from: https://github.com/facebookresearch/Detectron/blob/master/configs/04_2018_gn_baselines/scratch_e2e_mask_rcnn_R-101-FPN_3x_gn.yaml
        # https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("broccoli_amodal_train",)
        cfg.DATASETS.VAL = ("broccoli_amodal_val",)
        cfg.DATASETS.TEST = ("broccoli_amodal_test",)
        cfg.NUM_GPUS = 2
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.WEIGHT_DECAY = 0.0001
        cfg.SOLVER.LR_POLICY = 'steps_with_decay'
        cfg.SOLVER.BASE_LR = 0.02
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.MAX_ITER = 5000
        cfg.SOLVER.STEPS = (0, 4000, 4500)
        cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)
        cfg.OUTPUT_DIR = "weights/broccoli_amodal_temp"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) 

        #Start the training 
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
