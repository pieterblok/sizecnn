# SizeCNN - deep learning methods to size the occluded crop
![Size the invisible crop](./demo/20200819_143612133900_plant1188_rgb_trigger002.png?raw=true)
<br/>


## Summary
We provide two methods that can better estimate the size of occluded objects. Both methods use ORCNN (https://github.com/waiyulam/ORCNN), which is an extended Mask R-CNN network that outputs two masks for each object:
1. The regular visible mask ***(purple mask below)***
2. An additional amodal mask of the visible and invisible pixels ***(green mask below)*** <br/>
![Amodal_Visible_Masks](./demo/20200819_143612133900_plant1188_rgb_trigger002_amodal_visible_masks.png?raw=true)

## Installation
See [INSTALL.md](INSTALL.md)


## Getting started
We provide two methods that can be used to estimate the diameter of occluded crops: <br/>

1. [ORCNN_PostProcessing.md](ORCNN_PostProcessing.md) 
2. [ORCNN_Regression.md](ORCNN_Regression.md) 

To compare the sizing performance of both methods, we also provide also a "base-line" method (using Mask R-CNN & Post-processing): <br/>
1. [MRCNN_PostProcessing.md](MRCNN_PostProcessing.md)


## License
Our CNNs were forked from ORCNN (https://github.com/waiyulam/ORCNN), which was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, our CNNs will be released under the [Apache 2.0 license](LICENSE). <br/>


## Acknowledgements
The methods were developed by Pieter Blok (pieter.blok@wur.nl)
