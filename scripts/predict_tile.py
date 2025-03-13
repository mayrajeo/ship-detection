import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sahi_utils import *
from src.data_paths import *
from sahi.models.ultralytics import UltralyticsDetectionModel

from sahi.predict import get_sliced_prediction
import torch
from fastcore.script import *
from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely
from time import time

def get_longer_edge(geom:shapely.geometry.Polygon) -> float:
    x, y = shapely.geometry.box(*geom.bounds).exterior.coords.xy
    edge_lengths = (shapely.geometry.Point(x[0],y[0]).distance(shapely.geometry.Point(x[1],y[1])),
                    shapely.geometry.Point(x[1],y[1]).distance(shapely.geometry.Point(x[2],y[2])))
    return max(edge_lengths)

def clean_stationary_targets(gdf:gpd.GeoDataFrame, preset:str=None, keep_fps:bool=False) -> gpd.GeoDataFrame:
    "Run the postprocessing chain to get rid of obviously false predictions"
    # Clip predictions with topographical database lake and seawater, and stream area classes
    tot_bounds_3067 = tuple(gdf.to_crs('EPSG:3067').total_bounds)
    start = time()
    if preset is None:
        lakes = gpd.read_file(PATH_TO_WATERS, layer='jarvi', bbox=tot_bounds_3067).dissolve(by='kohdeluokka')
        seas = gpd.read_file(PATH_TO_WATERS, layer='meri', bbox=tot_bounds_3067).dissolve(by='kohdeluokka')
        rivers = gpd.read_file(PATH_TO_RIVERS, layer='virtavesialue', bbox=tot_bounds_3067).dissolve(by='kohdeluokka')
        lakes = lakes.to_crs(gdf.crs)
        seas = seas.to_crs(gdf.crs)
        rivers = rivers.to_crs(gdf.crs)
        waters = pd.concat([lakes, seas, rivers])
    elif preset == 'archipelago':
        waters = gpd.read_file(PATH_TO_ARCHI_WATERS, layer='waters')
    elif preset == 'gof':
        waters = gpd.read_file(PATH_TO_GOF_WATERS, layer='waters')
    elif preset in ['34VER', '34VEN']:
        waters = gpd.read_file(f'{PATH_TO_MTK_DATA}/{preset}_waters.gpkg', layer='waters').to_crs(gdf.crs)
    print('Cleaning predictions...')

    # Keep only predictions whose centroids are within sea, lake or a largeish river.
    print('Removing predictions that are not on water')

    # Add variables for logging
    rem = len(gdf)
    orig_len = len(gdf)
    n_vesikivikko = 0
    n_too_large = 0
    n_above_water = 0
    n_land = 0
    n_beacons = 0
    n_turbines = 0
    n_fisheries = 0
    fps = None
    
    # Set geometries to centroid points for filtering based on area
    gdf['centroids'] = gdf.centroid
    gdf.set_geometry('centroids', inplace=True)
    joined = gdf.sjoin(waters, how='inner', predicate='within')[['label', 'score', 'geometry', 'centroids']]
    fps = gdf[~gdf.index.isin(joined.index)].copy()
    fps['id'] = 'land'
    gdf = gdf[gdf.index.isin(joined.index)]
    n_land = rem - len(gdf)
    rem = len(gdf)
    # Filter large rock formations from topographical database MTK-muut_22-03-03 layer `vesikivikko`
    if rem > 0:
        print('Removing predictions that are on `vesikivikko`')
        rocks = gpd.read_file(PATH_TO_OTHER, layer='vesikivikko', bbox=tot_bounds_3067).dissolve(by='kohdeluokka')
        rocks = rocks.to_crs(gdf.crs)
        # Keep only predictions whose centroids are not within `vesikivikko`
        joined = gpd.sjoin(gdf, rocks, predicate='within', how='inner')
        if fps is None:
            fps = gdf[gdf.index.isin(joined.index)].copy()
            fps['id'] = 'rock'
        else:
            temp = gdf[gdf.index.isin(joined.index)].copy()
            temp['id'] = 'rock'
            fps = pd.concat([fps,temp])
        gdf = gdf[~gdf.index.isin(joined.index)].copy()
        n_vesikivikko = rem - len(gdf)
        rem = len(gdf)

    if rem > 0:
        print('Removing fisheries')
        # Keep only predictions whose centroids are not within fishery areas
        fisheries = gpd.read_file(PATH_TO_FISHERIES, bbox=tot_bounds_3067).to_crs(gdf.crs)
        joined = gpd.sjoin(gdf, fisheries, predicate='within', how='inner')
        if fps is None:
            fps = gdf[gdf.index.isin(joined.index)].copy()
            fps['id'] = 'fishery'
        else:
            temp = gdf[gdf.index.isin(joined.index)].copy()
            temp['id'] = 'fishery'
            fps = pd.concat([fps, temp])
        gdf = gdf[~gdf.index.isin(joined.index)]
        n_fisheries = rem - len(gdf)
        rem = len(gdf)

    # Revert geometry to bounding boxes and drop unnecessary columns
    gdf.set_geometry('geometry', inplace=True)
    gdf = gdf[['id', 'label', 'score', 'geometry']]
    if fps is not None:
        fps.set_geometry('geometry', inplace=True)
        fps = fps[['label', 'score', 'geometry', 'id']]
    # Filter above water rocks
    if rem > 0:
        print('Removing predictions that are rocks above waterline')
        above_water_rocks = gpd.read_file(PATH_TO_OTHER, layer='vesikivi', bbox=tot_bounds_3067).to_crs(gdf.crs)
        above_water_rocks = above_water_rocks[above_water_rocks.kohdeluokka.isin([38511,38512,38513])]
        if fps is None:
            fps = gdf.loc[(any(g.contains(above_water_rocks.geometry)) for g in gdf.geometry)].copy()
            fps['id'] = 'rocks'
        else:
            temp = gdf.loc[(any(g.contains(above_water_rocks.geometry)) for g in gdf.geometry)].copy()
            temp['id'] = 'rocks'
            fps = pd.concat([fps, temp])
        gdf = gdf.loc[(not any(g.contains(above_water_rocks.geometry)) for g in gdf.geometry)]

        n_above_water = rem - len(gdf)
        rem = len(gdf)

    # Filter Beacons etc.
    # Sector lights and lighthouses are such beacons that they might be visible from satellite images
    # Turvalaitteet from Väylävirasto and `ty_njr` classes 1, 2, 3 4, 5, 8
    # Description: https://ava.vaylapilvi.fi/ava/Vesi/Tietokuvaus/vesivayla-aineistot_tietosisallonkuvaus.pdf
    if rem > 0:
        print('Removing predictions that are either beacons or lighthouses')
        beacons = gpd.read_file(PATH_TO_BEACONS, tot_bounds_3067).to_crs(gdf.crs)
        beacons = beacons[beacons.ty_jnr.isin([1,2,3,4,5,8])]
        if fps is None:
            fps = gdf.loc[(any(g.contains(beacons.geometry)) for g in gdf.geometry)].copy()
            fps['id'] = 'beacon'
        else:
            temp = gdf.loc[(any(g.contains(beacons.geometry)) for g in gdf.geometry)].copy()
            temp['id'] = 'beacon'
            fps = pd.concat([fps, temp])
        gdf = gdf.loc[(not any(g.contains(beacons.geometry)) for g in gdf.geometry)]
        n_beacons = rem - len(gdf)
        rem = len(gdf)

    if rem > 0:
        # Filter wind turbines provided there is a layer for them
        print('Removing wind turbines')
        windmills = gpd.read_file(PATH_TO_BUILDINGS, layer='tuulivoimala', bbox=tot_bounds_3067).to_crs(gdf.crs)
        if fps is None:
            fps = gdf.loc[(any(g.contains(windmills.geometry)) for g in gdf.geometry)].copy()
            fps['id'] = 'windmill'
        else:
            temp = gdf.loc[(any(g.contains(windmills.geometry)) for g in gdf.geometry)].copy()
            temp['id'] = 'windmill'
            fps = pd.concat([fps, temp])
        gdf = gdf.loc[(not any(g.contains(windmills.geometry)) for g in gdf.geometry)]
        n_turbines = rem - len(gdf)
        rem = len(gdf)

    if rem > 0:
        # Finally remove predictions that have side longer than 750 meters.
        print('Removing predictions that are obviously too large')
        gdf['max_edge'] = gdf.geometry.apply(get_longer_edge)
        if fps is None:
            fps = gdf[gdf['max_edge'] > 750].copy()
            fps['id'] = 'size'
        else:
            temp = gdf[gdf['max_edge'] > 750].copy()
            temp['id'] = 'size'
            fps = pd.concat([fps, temp])
        gdf = gdf[gdf['max_edge'] <= 750]
        n_too_large = rem -len(gdf)

    end = time()
    print(f"""Summary of cleaning, took {end-start:.4f} seconds:
    Original number of predictions: {orig_len}
    Predictions on land: {n_land}
    Predictions on `vesikivikko`: {n_vesikivikko}
    Predictions that are fisheries: {n_fisheries}
    Predictions contained an above water rock: {n_above_water}
    Predictions contained a beacon etc: {n_beacons}
    Predictions contained a wind turbine: {n_turbines}
    Predictions larger than threshold: {n_too_large}
    """)

    if keep_fps:
        fps = fps[['id', 'label', 'score', 'geometry']]
        gdf = pd.concat([gdf, fps])
    return gdf

@call_parse
def main(yolov8_weights:str, # Path to yolov8 model weights to use
         tile:str, # Path to RGB S2_L1C tile to predict
         outpath:str, # Directory to save the results
         use_cuda:bool, # Whether to use cuda if it is available
         use_tta:bool, # Whether to use Test-time augmentation
         half:bool, # Whether to use half-precision 
         postproc:bool, # Whether to clean the predictions or not
         keep_fps:bool, # Whether to keep the cleaned fps in the results
         image_size:int=640, # Image size for YOLOv8 model
         slice_size:int=320, # Slice size to use with sahi
         conf_th:float=0.001, # Confidence threshold for predictions
         preset:str=None, # Area preset for faster filtering
    ):
    "Run YOLOv8 model with sahi for the full S2_L1C -tile and save predictions to `outpath`"

    outpath = Path(outpath)
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    print(f'Using {device} for predictions...')
    if half: print(f'Using half precision for inference')
    # Initialize model
    det_model = UltralyticsDetectionModel(model_path=yolov8_weights,
                                          device=device,
                                          confidence_threshold=conf_th,
                                          image_size=image_size)
    
    det_model.model.overrides.update({
        'augment': use_tta,
        'half': half,
        'conf': conf_th
    })

    # TODO Clean clouds from images. Either mask them out before predictions or do it afterwards. Shouldn't matter that much?

    # Get predictions
    sliced_pred_results = get_sliced_prediction(tile, 
                                                det_model, 
                                                slice_width=slice_size,
                                                slice_height=slice_size,
                                                overlap_height_ratio=0.2,
                                                overlap_width_ratio=0.2,
                                                perform_standard_pred=False,
                                                verbose=2)
    

    # Georeference predictions. Doesn't really need image but rather the affine object `im.transform` and crs `im.crs`
    tfmd_gdf = georef_sahi_preds(preds=sliced_pred_results.object_prediction_list,
                                 path_to_ref_img=tile)
    
    print(f'Found {len(tfmd_gdf)} objects before filtering')

    if len(tfmd_gdf) == 0: 
        print('No objects found')
        return

    tile_fn = tile.split('/')[-1].split('.')[0]

    # Fix labeling to start from 1 to work with COCO evaluation
    tfmd_gdf = tfmd_gdf[['label', 'score', 'geometry']]
    tfmd_gdf['id'] = 'boat'
    tfmd_gdf['label'] += 1

    if postproc: 
        tfmd_gdf = clean_stationary_targets(tfmd_gdf, preset=preset,keep_fps=keep_fps)

    print(f'{len(tfmd_gdf)} objects remain.')
    tfmd_gdf.to_file(outpath/f'{tile_fn}.geojson', driver='GeoJSON')
