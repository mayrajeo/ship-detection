from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from geo2ml.data.cv import shp_to_coco, shp_to_coco_results
from fastcore.basics import *
from pathlib import Path
import numpy as np
import os

class GisCOCOeval():
    
    def __init__(self, raster_path:Path, vector_path:Path, prediction_path:Path,
                 outpath:Path, coco_info:dict, coco_licenses:list, coco_categories:list):
        "Initialize evaluator with data path and coco information"
        store_attr()
        self.iou_threshs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
         
    def prepare_data(self, gt_label_col:str='label', res_label_col:str='label', 
                     rotated_bbox:bool=False, min_bbox_area:int=0):
        "Convert GIS-data predictions to COCO-format for evaluation, and save resulting files to self.outpath"
        print(os.path.isdir(self.prediction_path))
        shp_to_coco(raster_path=self.raster_path,
                    shp_path=self.vector_path,
                    outpath=self.outpath,
                    label_col=gt_label_col,
                    coco_categories=self.coco_categories,
                    coco_info=self.coco_info,
                    coco_licenses=self.coco_licenses,
                    min_bbox_area=min_bbox_area)
        shp_to_coco_results(prediction_path=self.prediction_path,
                            raster_path=self.raster_path,
                            coco_dict=self.outpath/'coco.json',
                            outfile=self.outpath/'coco_res.json',
                            label_col=res_label_col,
                            rotated_bbox=rotated_bbox)
    
    def prepare_eval(self, eval_type:str='segm'):
        """
        Prepare COCOeval to evaluate predictions with 100 and 1000 detections. AP metrics are evaluated with 1000 detections and AR with 100
        """
        self.coco = COCO(f'{self.outpath}/coco.json')
        self.coco_res = self.coco.loadRes(f'{self.outpath}/coco_res.json')
        self.coco_eval = COCOeval(self.coco, self.coco_res, eval_type)
        self.coco_eval.params.maxDets = [100, 1000]
        
    def evaluate(self, classes_separately:bool=True):
        "Run evaluation and print metrics"
        
        if classes_separately:
            for cat in self.coco_categories:
                print(f'\nEvaluating for category {cat["name"]}')
                self.coco_eval.params.catIds = [cat['id']]
                self.coco_eval.evaluate()
                self.coco_eval.accumulate()
                _summarize_coco(self.coco_eval)
        
        self.coco_eval.params.catIds = self.coco.getCatIds()
        print('\nEvaluating for full data...')

        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        _summarize_coco(self.coco_eval)
    
def _summarize_coco(cocoeval:COCOeval): 
    """
    Compute and display summary metrics for evaluation results.
    """
    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = cocoeval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = cocoeval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = cocoeval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1, maxDets=cocoeval.params.maxDets[1])
        stats[1] = _summarize(1, iouThr=.5, maxDets=cocoeval.params.maxDets[1])
        stats[2] = _summarize(1, iouThr=.75, maxDets=cocoeval.params.maxDets[1])
        stats[3] = _summarize(1, areaRng='small', maxDets=cocoeval.params.maxDets[1])
        stats[4] = _summarize(1, areaRng='medium', maxDets=cocoeval.params.maxDets[1])
        stats[5] = _summarize(1, areaRng='large', maxDets=cocoeval.params.maxDets[1])
        stats[6] = _summarize(0, maxDets=cocoeval.params.maxDets[0])
        stats[9] = _summarize(0, areaRng='small', maxDets=cocoeval.params.maxDets[0])
        stats[10] = _summarize(0, areaRng='medium', maxDets=cocoeval.params.maxDets[0])
        stats[11] = _summarize(0, areaRng='large', maxDets=cocoeval.params.maxDets[0])
        return stats
    
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    
    if not cocoeval.eval:
        raise Exception('Please run accumulate() first')
        
    iouType = cocoeval.params.iouType
    
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    cocoeval.stats = summarize()