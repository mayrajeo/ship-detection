import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import GisCOCOeval
from pathlib import Path
from fastcore.script import *

@call_parse
def evaluate_results(impath:Path, # Path to image folder
                     vector_path:Path, # Path to reference vector data
                     pred_path:Path, # Path to predictions
                     outpath:Path # Where to save the results
                     ):
    boat_categories = [{'supercategory':'boat', 'id':1, 'name':'boat'}]
    coco_eval = GisCOCOeval(raster_path=impath, 
                            vector_path=vector_path,
                            prediction_path=pred_path,
                            outpath=outpath, 
                            coco_info=None, 
                            coco_licenses=None, 
                            coco_categories=boat_categories)
    coco_eval.prepare_data(gt_label_col='id')
    coco_eval.prepare_eval(eval_type='bbox')
    coco_eval.coco_eval.params.maxDets = [1000, 1000]
    coco_eval.evaluate(classes_separately=False)
