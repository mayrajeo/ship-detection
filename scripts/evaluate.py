import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
import fiona
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from ultralytics.utils.metrics import DetMetrics
import json

from fastcore.script import *

def get_tp_fp_fn(pred_gdf:gpd.GeoDataFrame, gt_gdf:gpd.GeoDataFrame, 
                 iou_threshold:float, area_limit:gpd.GeoDataFrame=None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Derive true positives, false positives and false negatives

    Args:
        pred_gdf (geopandas.GeoDataFrame): Dataframe containing polygon predictions
        gt_gdf (geopandas.GeoDataFrame): Dataframe containing ground truth polygons
        iou_threshold (float): The intersection-over-union threshold to use for matching polygons.

    Returns:
        tuple: (tp, fp, fn) geodataframes containing the true positives, false positives and false negatives
    """

    # Copy the dataframes to avoid modifying the originals
    pred_gdf_copy = pred_gdf.copy()
    gt_gdf_copy = gt_gdf.copy()
    
    if area_limit: # Clip to only contain some subarea
        pred_gdf_copy = pred_gdf_copy.clip(area_limit)
        gt_gdf_copy = gt_gdf_copy.clip(area_limit)
    
    # Count the number of true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    tp_ixs = []
    fp_ixs = []
    for index, pred_row in pred_gdf_copy.iterrows():
        # Find the ground truth polygons that overlap with the prediction
        overlapping_gts = gt_gdf_copy[gt_gdf_copy.geometry.intersects(pred_row.geometry)]
        if overlapping_gts.empty:
            false_positives += 1
            fp_ixs.append(index)
        else:
            # Compute IoU between the prediction and overlapping ground truth polygons
            ious = overlapping_gts.geometry.intersection(pred_row.geometry).area / overlapping_gts.geometry.union(pred_row.geometry).area

            # Find the maximum IoU and corresponding index
            max_iou = np.max(ious.values)
            max_iou_index = np.argmax(ious.values)
            # If the maximum IoU is greater than the threshold, it's a true positive
            if max_iou >= iou_threshold:
                true_positives += 1
                tp_ixs.append(index)
                # Remove the matched ground truth polygon to avoid double counting
                gt_gdf_copy.drop(overlapping_gts.index[max_iou_index], inplace=True)
            else:
                false_positives += 1
                fp_ixs.append(index)

    # Any remaining ground truth polygons are false negatives
    false_negatives = gt_gdf_copy
    # Compute precision and recall
    return pred_gdf.iloc[tp_ixs], pred_gdf.iloc[fp_ixs], false_negatives


@call_parse
def evaluate(
    ground_truth_path:Path, # Path containing the ground truth geopackages
    result_dir:Path, # Path which contains the results
    outpath:Path, # Where to save the evalresults.json -file
):
    yolo_aps = [[],[],[],[],[],[],[],[],[],[]]
    yolo_precs =  [[],[],[],[],[],[],[],[],[],[]]
    yolo_recs =  [[],[],[],[],[],[],[],[],[],[]]
    ious = np.linspace(0.5, 0.95, 10)

    tiles = os.listdir(result_dir)
    for ix, i in enumerate(ious):
        tps_df = None
        fps_df = None
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        detmetrics = DetMetrics(names={1:'boat'})
        for t in tiles:
            tsteps = fiona.listlayers(ground_truth_path/f'{t.split(".")[0]}.gpkg')
            for tstep in tsteps:
                targs = gpd.read_file(ground_truth_path/f'{t}.gpkg', layer=tstep)
                preds = gpd.read_file(result_dir/t/f'{tstep}.geojson')
                tp, fp, fn = get_tp_fp_fn(preds, targs, i)
                if tps_df is None: tps_df = tp
                else: tps_df = pd.concat((tps_df, tp))
                if fps_df is None: fps_df = fp
                else: fps_df = pd.concat((fps_df, fp))
                tot_tp += len(tp)
                tot_fp += len(fp)
                tot_fn += len(fn)
        tp_ixs = np.ones((len(tps_df), 1))
        fp_ixs = np.zeros((len(fps_df),1))
        tps_for_yolo = np.concatenate((tp_ixs, fp_ixs))
        confs_for_yolo = np.concatenate((tps_df.score, fps_df.score))
        target_classes = np.ones_like(tps_for_yolo)[:,0]
        pred_classes = np.ones_like(tps_for_yolo)[:,0]
        detmetrics.process(tps_for_yolo, confs_for_yolo, target_classes, pred_classes)
        yolo_aps[ix] = detmetrics.box.ap[0]
        yolo_precs[ix] = tot_tp / (tot_tp + tot_fp)
        yolo_recs[ix] = tot_tp / (tot_tp + tot_fn)
    
    outdict = {
        'precision': yolo_precs[0],
        'recall': yolo_recs[0],
        'mAP50': yolo_aps[0],
        'mAP50-95': np.mean(yolo_aps)
    }

    with open(outpath/'evalresults.json', 'w') as dest:
        json.dump(outdict, dest, sort_keys=True, indent=4)