# Scripts

For each script, running `python <script> -h` provides documentation and description.

## cdse_odata.py

Utility to download complete S2 products from Copernicus Data Space Ecosystem odata api. 

## cdse_odata_tci.py

Utility to download only TCI file of a product from Copernicus Data Space Ecosystem odata api. 

## make_yolo_data.py

Script to convert the dataset into YOLO-format dataset.

## get_tile_list.py

Queries CDSE odata api for available products for `tile_id` between `start_date` and `end_date`, with maximum `cloud_cover`.
 
## train_models.py, train_revision_models.py

Main training scripts for the models. `train_revision_models.py` is the latest version.

## predict_tile.py

Runs model for large mosaics, and return the predictions as a geojson file. Also optionally post-processes the predictions by marking stationary targets and detections outside of water areas, optionally deleting them from the predictions.

Assumes that you have a file `src/data_paths.py` which contain the locations for external datasets, such as Topographic database as geopackage files and Maritime transport, Aids to navigation as a shapefile. 

## evaluate.py

Evaluate models based on predictions for full tiles, and outputs COCO metrics with ultralytics implementations.