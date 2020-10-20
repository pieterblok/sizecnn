# SizeCNN - a deep learning method to size the invisible crop
![Size the invisible crop](./demo/20200819_143612133900_plant1188_rgb_trigger002.png?raw=true)


## Installation

See [INSTALL.md](INSTALL.md).


## Features

SizeCNN consists of two CNN's: 
<br/>
1. ORCNN (https://github.com/waiyulam/ORCNN), which is an extended Mask R-CNN network that outputs two masks for each object:
   1. The regular visible mask ***(purple mask below)***
   2. An additional amodal mask of the visible and invisible pixels ***(green mask below)***
![Amodal_Visible_Masks](./demo/20200819_143612133900_plant1188_rgb_trigger002_amodal_visible_masks.png?raw=true)
2. A CNN regression network that estimates the diameter (mm) from the registered XYZ image, using the visible and the amodal mask


## License
**Code developed by Pieter Blok (pieter.blok@wur.nl)**
<br/>

Cite ORCNN:
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
