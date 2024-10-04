# Detecting marine vessels from Sentinel-2 imagery with YOLOv8

Code and documentation repository for Detecting marine vessels from Sentinel-2 imagery using YOLOv8 object detection framework. 

Example app: [https://huggingface.co/spaces/mayrajeo/marine-vessel-detection](https://huggingface.co/spaces/mayrajeo/marine-vessel-detection). 

## Getting started

### Installation

Install required environment `conda env create -f torch2-env.yml`

### Data

Models are trained on Sentinel-2 RGB images from June, July and August. Data consist of five separate Sentinel-2 tiles from the Finnish coast, with three separate acquisitions from each. Products were downloaded as L1C-products, and color-corrected using same protocol as in Tarkka+ service by Finnish Environment Institute.

Reference data were manually annotated by comparing three separate acquisitions and drawing a bounding box around detected marine vessel. The datasets are available on Zenodo portal: [https://zenodo.org/records/10046342](https://zenodo.org/records/10046342). 

### Models

Model weights and config files with best mAP50 score for each YOLOv8 model architecture are available on [https://huggingface.co/mayrajeo/marine-vessel-detection](https://huggingface.co/mayrajeo/marine-vessel-detection). 