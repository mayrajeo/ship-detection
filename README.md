# Detecting marine vessels from Sentinel-2 imagery with YOLO object detection models

Code and documentation repository for Detecting marine vessels from Sentinel-2 imagery using ultralytics object detection framework. 

## Getting started

### Installation

Install required environment `conda env create -f ship-env.yml`

### Data

Models are trained on Sentinel-2 images, consisting of five separate Sentinel-2 tiles from the Finnish coast, with three or four separate acquisitions from each. Products were downloaded as L1C-products from Copernicus Data Space Ecosystem.

Reference data were manually annotated by comparing several separate acquisitions and drawing a bounding box around detected marine vessel. The datasets are available on Zenodo portal: [10.5281/zenodo.10046341](https://doi.org/10.5281/zenodo.10046341). 

### Models

Model checkpoints are available on [https://huggingface.co/mayrajeo/marine-vessel-yolo](https://huggingface.co/mayrajeo/marine-vessel-yolo). Currently the model with the best test dataset performance is avalable, which is trained on L1C-TCI data.  