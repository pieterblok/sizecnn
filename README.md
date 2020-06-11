# HarvestCNN 
**Developed by Pieter Blok (pieter.blok@wur.nl)**
<br/>
**HarvestCNN uses ORCNN/Detectron2 (https://github.com/waiyulam/ORCNN) and an additional depth regression**


## Installation

See [INSTALL.md](INSTALL.md).


## Run the demo code

- download the pretrained coco weights: 
https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl
- place the weights-file in a folder called weights in the harvestcnn root directory
- cd harvestcnn
- cd demo
- python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input ../demo/testImage20150714_084518730.jpg --opts MODEL.WEIGHTS ../weights/model_final_2d9806.pkl
- Press ESC to close the window


## Citing ORCNN

```BibTeX
@article{DBLP:journals/corr/abs-1804-08864,
  author    = {Patrick Follmann and
               Rebecca K{\"{o}}nig and
               Philipp H{\"{a}}rtinger and
               Michael Klostermann},
  title     = {Learning to See the Invisible: End-to-End Trainable Amodal Instance
               Segmentation},
  journal   = {CoRR},
  volume    = {abs/1804.08864},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.08864},
  archivePrefix = {arXiv},
  eprint    = {1804.08864},
  timestamp = {Mon, 13 Aug 2018 16:46:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1804-08864.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
