# SizeCNN - a deep learning method to size the invisible crop
![Size the invisible crop](./demo/20200819_143612133900_plant1188_rgb_trigger002.png?raw=true)


## Features

SizeCNN consists of two CNN's: 
<br/>
1. ORCNN (https://github.com/waiyulam/ORCNN), which is an extended Mask R-CNN network that outputs two masks for each object:
   1. The regular visible mask ***(purple mask below)***
   2. An additional amodal mask of the visible and invisible pixels ***(green mask below)***
![Amodal_Visible_Masks](./demo/20200819_143612133900_plant1188_rgb_trigger002_amodal_visible_masks.png?raw=true)
2. A CNN regression network that estimates the diameter (mm) from the registered XYZ image, using the visible and the amodal mask


## Installation

See [INSTALL.md](INSTALL.md).


## Getting started
1. Annotate the dataset, see [ANNOTATE.md](ANNOTATE.md)
2. Train ORCNN, see [ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb](ORCNN_Train_and_Evaluate_AmodalVisibleMasks.ipynb)
3. Train the CNN regression network, see 
4. Infer the dataset using the trained ORCNN and regression network, see [Diameter_regression_AmodalVisibleMasks.ipynb](Diameter_regression_AmodalVisibleMasks.ipynb)


## License
**Code developed by Pieter Blok (pieter.blok@wur.nl)**
<br/>
