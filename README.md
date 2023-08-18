# Detecting marine vessels from Sentinel-2 imagery with YOLOv8

Code and documentation repository for Detecting marine vessels from Sentinel-2 imagery using YOLOv8 object detection framework. 

Example app running on [https://huggingface.co/spaces/mayrajeo/marine-vessel-detection](https://huggingface.co/spaces/mayrajeo/marine-vessel-detection). 

## Getting started

### Installation

TBA

### Data

Models are trained on Sentinel-2 RGB images from June, July and August. Data consist of five separate Sentinel-2 tiles with three separate acquisitions from each. Products were downloaded as L1C-products, and color-corrected using same protocol as in Tarkka+ service by Finnish Environment Institute.

Reference data were manually annotated by comparing three separate acquisitions and drawing a bounding box around detected marine vessel. 

### Models

Model weights with best mAP50 score for each YOLOv8 model type are available on [https://huggingface.co/mayrajeo/marine-vessel-detection]. 