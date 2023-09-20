import geopandas as gpd 
from shapely.geometry import box, Polygon, MultiPolygon
from geo2ml.data.coordinates import *

def georef_sahi_preds(preds, path_to_ref_img, result_type='bbox') -> gpd.GeoDataFrame:
    "Converts a list of `ObjectPredictions` to a geodataframe, georeferenced according to reference image"
    labels = [p.category.id for p in preds]

    if result_type == 'bbox': 
        polys = [box(*p.bbox.to_xyxy()) for p in preds]
    elif result_type == 'mask': 
        polys = []
        for p in preds:
            segmentation = p.mask.segmentation
            temp_polys = []
            for segm in segmentation:
                xy_coords = [(segm[i], segm[i+1]) for i in range(0, len(segm), 2)]
                xy_coords.append(xy_coords[-1])
                temp_polys.append(Polygon(xy_coords))
            polys.append(MultiPolygon(temp_polys))
    else:
        print(f'Unknown result type {result_type}, defaulting to bbox')
        polys = [box(*p.bbox.to_xyxy()) for p in preds]
    scores = [p.score.value for p in preds]
    gdf = gpd.GeoDataFrame({'label':labels, 'geometry':polys, 'score':scores})
    tfmd_gdf = georegister_px_df(gdf, path_to_ref_img)
    return tfmd_gdf