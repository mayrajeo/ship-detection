import geopandas as gpd 
from shapely.geometry import box
from drone_detector.processing.coordinates import *

def georef_sahi_preds(preds, path_to_ref_img) -> gpd.GeoDataFrame:
    "Converts a list of `ObjectPredictions` to a geodataframe, georeferenced according to reference image"
    labels = [p.category.id for p in preds]
    polys = [box(*p.bbox.to_xyxy()) for p in preds]
    scores = [p.score.value for p in preds]
    gdf = gpd.GeoDataFrame({'label':labels, 'geometry':polys, 'score':scores})
    tfmd_gdf = georegister_px_df(gdf, path_to_ref_img)
    return tfmd_gdf