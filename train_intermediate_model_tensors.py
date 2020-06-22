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

# class checkpoint: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/checkpoint.py
import copy
import logging
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
import torch.nn as nn
from fvcore.common.file_io import PathManager
from torch.nn.parallel import DataParallel, DistributedDataParallel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog,MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.engine import ModalMaskHeadTrainer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils import comm


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


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


# thanks to: https://github.com/facebookresearch/fvcore/blob/7bd0aeeb7292eb168abf7139a804bf04dc17b994/fvcore/common/checkpoint.py
class _IncompatibleKeys(
    NamedTuple(
        # pyre-fixme[10]: Name `IncompatibleKeys` is used but not defined.
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: object,
    ) -> None:
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)  # pyre-ignore
        self.logger = logging.getLogger(__name__)  # pyre-ignore
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name: str, **kwargs: Dict[str, str]) -> None:
        """
        Dump model and checkpointables to a file.
        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str, checkpointables: Optional[List[str]] = None) -> object:
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.
        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))
        if not os.path.isfile(path):
            path = PathManager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:  # pyre-ignore
                self.logger.info("Loading {} from {}".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))  # pyre-ignore

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return PathManager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)  # pyre-ignore

    def get_all_checkpoint_files(self) -> List[str]:
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in PathManager.ls(self.save_dir)
            if PathManager.isfile(os.path.join(self.save_dir, file))
            and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True) -> object:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.
        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.
        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.
        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with PathManager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore

    def _load_file(self, f: str) -> object:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.
        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:  # pyre-ignore
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)
            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Unable to load '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model!".format(k, shape_checkpoint, shape_model)
            )
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(
                self.model, incompatible.missing_keys
            )
            if missing_keys:
                self.logger.info(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)

def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def write_metrics(metrics_dict: dict, storage):
    """
    Args:
        metrics_dict (dict): dict of scalar metrics
    """
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    # gather metrics among all workers for logging
    # This assumes we do DDP-style training, which is currently the only
    # supported method in detectron2.
    all_metrics_dict = comm.gather(metrics_dict)

    if comm.is_main_process():
        if "data_time" in all_metrics_dict[0]:
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())

        storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            storage.put_scalars(**metrics_dict)

    return metrics_dict


if __name__ == "__main__":
    # run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    if torch.cuda.is_available():
        # this might be needed for a proper definition of the test classes
        # register the amodal datasets 
        register_coco_instances("broccoli_amodal_train", {}, "datasets/broccoli_amodal/train/annotations.json", "datasets/broccoli_amodal/train")
        register_coco_instances("broccoli_amodal_val", {}, "datasets/broccoli_amodal/val/annotations.json", "datasets/broccoli_amodal/val")

        # create the metadata files 
        broccoli_amodal_train_metadata = MetadataCatalog.get("broccoli_amodal_train")
        broccoli_amodal_val_metadata = MetadataCatalog.get("broccoli_amodal_val")

        # create the dataset dicts 
        dataset_dicts_train = DatasetCatalog.get("broccoli_amodal_train")
        dataset_dicts_val = DatasetCatalog.get("broccoli_amodal_val")

        # configure the inference procedure
        # resuming from previous checkpoints (5000)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("broccoli_amodal_train",)
        cfg.DATASETS.VAL = ("broccoli_amodal_val",)
        cfg.NUM_GPUS = 2
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.WEIGHT_DECAY = 0.0001
        cfg.SOLVER.LR_POLICY = 'steps_with_decay'
        cfg.SOLVER.BASE_LR = 0.02
        # cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.GAMMA = 0.1
        cfg.SOLVER.CHECKPOINT_PERIOD = 5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (broccoli)

        # train from scratch:
        # cfg.SOLVER.WARMUP_ITERS = 1000
        # cfg.SOLVER.MAX_ITER = 5000
        # cfg.SOLVER.STEPS = (0, 4000, 4500)
        # cfg.SOLVER.WARMUP_ITERS = 50
        # cfg.SOLVER.MAX_ITER = 100
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        # cfg.OUTPUT_DIR = "weights/broccoli_amodal_temp2"
        # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) 

        # finetune from loaded weights:
        # probably the learning rate has to be decreased
        # cfg.SOLVER.MAX_ITER = 6000
        cfg.SOLVER.STEPS = (5000, 5100, 5500)
        # cfg.OUTPUT_DIR = "weights/broccoli_amodal_temp1"
        # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")      

        #Start the training from scratch
        #trainer = DefaultTrainer(cfg) 
        # trainer = ModalMaskHeadTrainer(cfg)
        # trainer.resume_or_load(resume=False)
        # trainer.train()

        #Resume the training from the loaded weights
        #trainer = DefaultTrainer(cfg) 
        # trainer = ModalMaskHeadTrainer(cfg)
        # trainer.resume_or_load(resume=True) # resume the training when there's a file called "last_checkpoint" is present in the cfg.OUTPUT_DIR
        # trainer.train()

        # load tensors with Pytorch
        # loadedweights = torch.load('/home/pieterdeeplearn/harvestcnn/weights/broccoli_amodal/model_final.pth', map_location=lambda storage, loc: storage.cuda(1))
        # print("stop")
      
        model = build_model(cfg)
        # thanks to: https://discuss.pytorch.org/t/how-to-freeze-the-part-of-the-model/31409
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False
            if "proposal_generator" in name:
                p.requires_grad = False
            if "roi_heads" in name:
                p.requires_grad = False
                if "visible_mask_head" in name and "invisible_mask_head" not in name:
                    p.requires_grad = True

        print("The following network layers will be trained:")
        for name, p in model.named_parameters():
            if p.requires_grad == True:
                print(name)

        process = ProcessImage(model, cfg)
        optimizer = build_optimizer(cfg, model)

        data_loader = build_detection_train_loader(cfg)
        data_loader_iter = iter(data_loader)
        
        start_iter = 5000
        max_iter = 6000    

        lowest_loss = 9999
        Ckp = Checkpointer(model, save_dir = "/home/pieterdeeplearn/harvestcnn/weights/broccoli_amodal_temp5")
        Ckp.resume_or_load(path = "/home/pieterdeeplearn/harvestcnn/weights/broccoli_amodal_temp1/model_final.pth", resume=True)

        with EventStorage(start_iter) as storage:
            for iter in range(start_iter, max_iter):
                start = time.perf_counter()
                data = next(data_loader_iter)
                data_time = time.perf_counter() - start

                images = process.preprocess_image(data)
                if "instances" in data[0]:
                    gt_instances = [x["instances"].to(torch.device(cfg.MODEL.DEVICE)) for x in data]

                # features = model.backbone(images.tensor)

                features1 = model.backbone(images.tensor)
                features2 = model.backbone(images.tensor)
                
                features_p2 = features1['p2'].add(features2['p2'])
                features_p3 = features1['p3'].add(features2['p3'])
                features_p4 = features1['p4'].add(features2['p4'])
                features_p5 = features1['p5'].add(features2['p5'])
                features_p6 = features1['p6'].add(features2['p6'])

                features = {'p2': features_p2, 'p3': features_p3, 'p4': features_p4, 'p5': features_p5, 'p6': features_p6}

                proposals, proposal_losses = model.proposal_generator(images, features, gt_instances)
                _, detector_losses = model.roi_heads(images, features, proposals, gt_instances)

                loss = {}
                loss.update(detector_losses)
                loss.update(proposal_losses)
                losses = sum(loss.values())

                metrics_dict = loss
                metrics_dict["data_time"] = data_time
                output_metrics = write_metrics(metrics_dict, storage)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if iter%20 == 0:
                    print(iter)
                    loss_print = losses.to("cpu").detach().numpy()
                    print(loss_print)
                    print(output_metrics)   

                    if loss_print < lowest_loss:
                        lowest_loss = loss_print
                        print("lowest loss set at: " + str(lowest_loss))
                        Ckp.save("model_final")                 

                storage.step()

        print("finished")