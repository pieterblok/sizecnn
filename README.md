# A deep learning method to size crops that are occluded
![Size the invisible crop](./demo/20200819_143612133900_plant1188_rgb_trigger002.png?raw=true)
<br/>


## Summary
We provide a deep-learning method to better estimate the size of occluded objects. The method is based on ORCNN (https://github.com/waiyulam/ORCNN), which is an extended Mask R-CNN network that outputs two masks for each object:
1. The regular visible mask ***(purple mask below)***
2. An additional amodal mask of the visible and invisible pixels ***(green mask below)*** <br/>
![Amodal_Visible_Masks](./demo/20200819_143612133900_plant1188_rgb_trigger002_amodal_visible_masks.png?raw=true)

## Installation
See [INSTALL.md](INSTALL.md)


## Getting started
The deep-learning method that can be used to estimate the diameter of occluded crops: <br/>
[ORCNN.md](ORCNN.md) 

The "base-line" method, which is based on Mask R-CNN and a circle fit method. This method can be compared to the ORCNN sizing method: <br/>
[MRCNN.md](MRCNN.md) 

## Results
We evaluated the sizing performance of the two methods on an independent test set of 487 RGB-D images. The broccoli heads in the test set had occlusion rates between 0% and 100%.

The table and the graph below summarizes the average absolute diameter error (mm) for 10 occlusion rates. The number between the brackets is the standard deviation (mm).
 
| Occlusion rate     | Mask R-CNN			| ORCNN 			| P-value Wilcoxon test		|
|--------------------|:--------------------------------:|:-----------------------------:|:-----------------------------:|
| 0.0 - 0.1 (n=147)  |  3.6 (3.1)       		| 4.0 (2.9)       		| 0.10 (ns)			|
| 0.1 - 0.2 (n=60)   |  3.2 (2.6)       		| 3.9 (2.4)       		| 0.06 (ns)			|
| 0.2 - 0.3 (n=33)   |  5.3 (4.1)       		| 5.4 (4.0)       		| 0.64 (ns)			|
| 0.3 - 0.4 (n=35)   |  7.0 (4.8)       		| 6.1 (4.5)       		| 0.39 (ns)			|
| 0.4 - 0.5 (n=48)   |  8.6 (7.0)       		| 6.4 (4.7)       		| 0.09 (ns)			|
| 0.5 - 0.6 (n=35)   |  10.6 (7.8)       		| 6.6 (6.0)       		| 0.02 (*)			|
| 0.6 - 0.7 (n=64)   |  16.5 (13.6)       		| 7.8 (7.8)       		| 0.00 (****)			|
| 0.7 - 0.8 (n=42)   |  25.2 (18.4)       		| 12.5 (10.2)       		| 0.00 (***)			|
| 0.8 - 0.9 (n=19)   |  44.1 (24.0)       		| 14.1 (13.7)      		| 0.00 (***)			|
| 0.9 - 1.0 (n=4)    |  77.2 (43.2)       		| 27.0 (27.5)      		| -				|
| All (n=487)        |  10.7 (15.3)       		| 6.5 (7.3)      		| 0.00 (****)			|

\- : too few samples, ns : P> 0.05, \* : 0.01 < P <= 0.05, \*\* : 0.01 < P <= 0.05, \*\*\* : 0.001 < P <= 0.01, \*\*\*\* : P <= 0.001
                            
![error_curve](./utils/diameter_error_occlusion_rate.jpg?raw=true)

## Dataset
We will upload our dataset when our publication has being published. The dataset consists of 1613 RGB-D images of broccoli heads with various occlusion rates. 

## Pretrained weights

| Network     | Backbone         		| Dataset  | Weights													|
| ------------|---------------------------------|----------|------------------------------------------------------------------------------------------------------------| 
| Mask R-CNN  | ResNext_101_32x8d_FPN_3x	| Broccoli | [model_0008999.pth](https://drive.google.com/file/d/14ruTcox7nPSBPxPPaYjETizJvS77mjVG/view?usp=sharing) 	|
| ORCNN	      | ResNext_101_32x8d_FPN_3x	| Broccoli | [model_0007999.pth](https://drive.google.com/file/d/1q7elXawUTw-ThZ2b3BHIOoZrmBZiLoMG/view?usp=sharing) 	|	


## License
Our software was forked from ORCNN (https://github.com/waiyulam/ORCNN), which was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, our CNNs will be released under the [Apache 2.0 license](LICENSE). <br/>


## Acknowledgements
The methods were developed by Pieter Blok (pieter.blok@wur.nl)
