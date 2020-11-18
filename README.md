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


## Results
We evaluated the sizing performance of the three methods on an independent test set of 487 RGB-D images. The broccoli heads in the test set had occlusion rates between 0% and 100%.

The table and the graph below summarizes the average absolute diameter error (mm) for 10 occlusion rates. The number between the brackets is the standard deviation (mm).
 
| Occlusion rate     | Mask R-CNN & Post-processing	| ORCNN & Post-processing	| ORCNN & Regression		|
|:------------------:|:--------------------------------:|:-----------------------------:|:-----------------------------:|
| 0.0 - 0.1          |  3.6 (3.1)       		| 4.0 (2.9)       		| 4.8 (3.7)			|
| 0.1 - 0.2          |  3.2 (2.6)       		| 3.9 (2.4)       		| 5.9 (4.7)			|
| 0.2 - 0.3          |  5.3 (4.1)       		| 5.4 (4.0)       		| 7.9 (4.6)			|
| 0.3 - 0.4          |  7.0 (4.8)       		| 6.1 (4.5)       		| 6.1 (5.1)			|
| 0.4 - 0.5          |  8.6 (7.0)       		| 6.4 (4.7)       		| 7.4 (6.6)			|
| 0.5 - 0.6          |  10.6 (7.8)       		| 6.6 (6.0)       		| 6.8 (5.4)			|
| 0.6 - 0.7          |  16.5 (13.6)       		| 7.8 (7.8)       		| 8.6 (7.4)			|
| 0.7 - 0.8          |  25.2 (18.4)       		| 12.5 (10.2)       		| 10.2 (8.7)			|
| 0.8 - 0.9          |  44.1 (24.0)       		| 14.1 (13.7)      		| 12.2 (11.5)			|
| 0.9 - 1.0          |  77.2 (43.2)       		| 27.0 (27.5)      		| 25.0 (15.7)			|
                            
![error_curve](./utils/diameter_error_occlusion_rate_three_methods.png?raw=true)

## Dataset
Our dataset will be uploaded soon. The dataset consists of 1613 RGB-D images of broccoli heads with various occlusion rates. 

## Pretrained weights

| Network	      	| Dataset         		| Weights													|
| ----------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------| 
| Mask R-CNN		| Broccoli			| [model_0008999.pth](https://drive.google.com/file/d/14ruTcox7nPSBPxPPaYjETizJvS77mjVG/view?usp=sharing) 	|
| ORCNN			| Broccoli			| [model_0007999.pth](https://drive.google.com/file/d/1q7elXawUTw-ThZ2b3BHIOoZrmBZiLoMG/view?usp=sharing) 	|
| CNN Regression	| Broccoli			| [epoch_056.pt](https://drive.google.com/file/d/1-hfNOvu0yZNavE2Zlo9XAMkdRQ0YKgyP/view?usp=sharing) 		|	


## License
Our CNNs were forked from ORCNN (https://github.com/waiyulam/ORCNN), which was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, our CNNs will be released under the [Apache 2.0 license](LICENSE). <br/>


## Acknowledgements
The methods were developed by Pieter Blok (pieter.blok@wur.nl)
