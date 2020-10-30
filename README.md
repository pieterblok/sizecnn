# HarvestCNN - a deep learning method to size the invisible crop
![Size the invisible crop](./demo/20200819_143612133900_plant1188_rgb_trigger002.png?raw=true)
<br/>


## Features
HarvestCNN consists of two CNN's: 
1. ORCNN (https://github.com/waiyulam/ORCNN), which is an extended Mask R-CNN network that outputs two masks for each object:
   1. The regular visible mask ***(purple mask below)***
   2. An additional amodal mask of the visible and invisible pixels ***(green mask below)*** <br/>
![Amodal_Visible_Masks](./demo/20200819_143612133900_plant1188_rgb_trigger002_amodal_visible_masks.png?raw=true)
2. A CNN regression network that estimates the diameter (mm) from the registered XYZ image, using the visible and the amodal mask


## Installation
See [INSTALL.md](INSTALL.md)


## Getting started
1. Annotate the dataset, see [ANNOTATE.md](ANNOTATE.md)
2. Train ORCNN, see [ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb](ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb)
3. Prepare an image dataset for CNN regression training, see [Create_XYZA_images_for_regression_training.ipynb](Create_XYZA_images_for_regression_training.ipynb)
4. Train the CNN regression network, see 
5. Estimate the diameters with the trained ORCNN and CNN regression network, see [Diameter_regression_AmodalVisibleMasks.ipynb](Diameter_regression_AmodalVisibleMasks.ipynb)


## Other methods to estimate the diameter
Besides HarvestCNN, it's also possible to estimate the diameter with two other methods:
1. [MaskRCNN_Filtering.md](MaskRCNN_Filtering.md) <br/>
2. [ORCNN_Filtering.md](ORCNN_Filtering.md) <br/>


## License
HarvestCNN was forked from ORCNN (https://github.com/waiyulam/ORCNN), which was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, HarvestCNN is released under the [Apache 2.0 license](LICENSE). <br/>


## Acknowledgements
HarvestCNN and the two other methods are developed by Pieter Blok (pieter.blok@wur.nl)
