# Mapping recreational marine traffic from Sentinel-2 imagery using YOLO object detection models

Code and documentation repository for article Mapping recreational marine traffic from Sentinel-2 imagery using YOLO object detection models, published in Remote Sensing of Environment: [https://doi.org/10.1016/j.rse.2025.114791](https://doi.org/10.1016/j.rse.2025.114791).

## Getting started

### Installation

Install required environment `conda env create -f ship-env.yml`

### Data

Reference dataset consist of five separate Sentinel-2 tiles from the Finnish coast, with three or four separate acquisitions from each. Products were downloaded as L1C-products from Copernicus Data Space Ecosystem.

Reference data were manually annotated by comparing several separate acquisitions and drawing a bounding box around detected marine vessel. The datasets are available on Zenodo portal: [10.5281/zenodo.10046341](https://doi.org/10.5281/zenodo.10046341). 

### Models

Model checkpoints are available on [https://huggingface.co/mayrajeo/marine-vessel-yolo](https://huggingface.co/mayrajeo/marine-vessel-yolo). Currently the model with the best test dataset performance is avalable. This checkpoints was trained on L1C-TCI data.

### Citation

> Mäyrä, J., Virtanen, E. A., Jokinen, A.-P., Koskikala, J., Väkevä, S., & Attila, J. (2025). Mapping recreational marine traffic from sentinel-2 imagery using Yolo Object Detection Models. Remote Sensing of Environment, 326, 114791. https://doi.org/10.1016/j.rse.2025.114791

```
@article{mayraMappingRecreational2025,
title = {Mapping recreational marine traffic from Sentinel-2 imagery using YOLO object detection models},
journal = {Remote Sensing of Environment},
volume = {326},
pages = {114791},
year = {2025},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2025.114791},
url = {https://www.sciencedirect.com/science/article/pii/S0034425725001956},
author = {Janne Mäyrä and Elina A. Virtanen and Ari-Pekka Jokinen and Joni Koskikala and Sakari Väkevä and Jenni Attila},
keywords = {Marine vessel detection, Object detection, Satellite imagery, Deep learning, Human pressures},
}
```

### Acknowledgments

This work was supported by Enhancing the marine and coastal biodiversity of the Baltic Sea in Finland and promoting the sustainable use of marine resources (LIFE-IP BIODIVERSEA (LIFE20 IPE/FI/000020)).

The authors wish to acknowledge CSC – IT Center for Science, Finland, for computational resources.