from fastcore.script import *
from ultralytics import YOLO, RTDETR
from pathlib import Path
import os
from ultralytics import settings
settings.update({"wandb": True})
import wandb
from shutil import rmtree

import torch
import numpy as np
from fastcore.basics import *
from ultralytics.models.yolo.detect.val import DetectionValidator

@patch_to(DetectionValidator)
def get_stats(self):
    """Returns metrics statistics and results dictionary."""
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
    self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
    self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
    stats.pop("target_img", None)
    if len(stats) and stats["tp"].any():
        self.metrics.process(**stats)
        return self.metrics.results_dict
    else:
        return {k: 0 for k in self.metrics.results_dict.keys()}

yolo_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', #  YOLOv8
               'yolov9t', 'yolov9s', 'yolov9m', 'yolov9c', 'yolov9e', # YOLOv9 
               'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x', # YOLOv10
               'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x', # YOLOv11
               'rtdetr-l', 'rtdetr-x' # RT-DETR
               ]

optimizers = ['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp']

@call_parse
def train_model(base_model:Param(help='Base model to use', 
                                 type=str, 
                                 choices=yolo_models),
                dataset_file:Param(help='Path to dataset yaml file', 
                                   type=Path),
                base_model_dir:Param(help='Folder for pretrained models', 
                                     type=Path),
                outdir:Param(help='Path to save the models', 
                             type=Path, 
                             default='runs'),
                epochs:Param(help='How many epochs to train', 
                             type=int, 
                             default=50), 
                patience:Param(help='Epochs to wait for no improvement for early stopping', 
                               type=int, 
                               default=10),
                batch:Param(help="""Batch size to use. 
                                 positive int for specific number of images,
                                 -1 for BS utilizing 60% of GPU memory""", 
                            default=-1,
                            type=int), 
                imgsz:Param(help='Image size to use', 
                            type=int, 
                            default=640),
                optimizer:Param(help='Optimizer to use', 
                                type=str, 
                                default='auto', 
                                choices=optimizers),
                cos_lr:Param(help='Whether to use cosine annealing', 
                             type=bool, 
                             default=False),
                amp:Param(help='Use AMP training?',
                          type=bool,
                          default=False),
                pretrained:Param(help='Use pretrained models?', 
                                 type=bool, 
                                 default=False),
                save_period:Param(help='save period for intermediate saving. -1 disables',
                                  type=int,
                                  default=-1),
                wandb_tags:Param(help='Tags for WandB, separated with ,', 
                                 type=str, 
                                 opt=True,
                                 default='none')
                ):

    outdir = Path(os.path.abspath(outdir))
    base_model_dir = Path(os.path.abspath(base_model_dir))
    dataset_file = Path(os.path.abspath(dataset_file))

    
    # Replace model path and project folder with your own
    model_file = base_model_dir/f'{base_model}.pt' if pretrained else f'{base_model}.yaml'

    optimizer_params = {
        'auto': {
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0
        },
        'SGD': {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005
        },
        'Adam': {
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0
        },
        'AdamW': {
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-3
        },
        'RMSProp': {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.0,
            'weight_decay': 0.0
        }
    }

    suffix = 'pretrained' if pretrained else 'scratch'

    project = outdir/f'{base_model}_{optimizer}'
    name = f'{dataset_file.parent.name}_{dataset_file.stem}_{suffix}'
    print(project/name)
    if os.path.exists(project/name):
        print(project/name)
        rmtree(project/name)

    opt = optimizer_params[optimizer]
    print(f'Starting to train with parameters {base_model} {optimizer} {dataset_file.stem}')
    wandb.init(project='ultralytics_marinevessels_revision', 
               job_type='training', 
               name=f'{base_model}_{dataset_file.parent.name}_{dataset_file.stem}_{"pt" if pretrained else "scratch"}',
               tags=None if wandb_tags == 'none' else wandb_tags.split(','))
    if base_model in ['rtdetr-l', 'rtdetr-x']: model = RTDETR(model_file)
    else: model = YOLO(model_file)
    results = model.train(data=dataset_file,
                          epochs=epochs,
                          patience=patience,
                          imgsz=imgsz,
                          batch=batch,
                          project=project,
                          name=name,
                          save=True,
                          save_period=save_period,
                          pretrained=pretrained,
                          cache='disk',
                          single_cls=True,
                          exist_ok=True,
                          amp=amp,
                          optimizer=optimizer,
                          cos_lr=cos_lr,
                          device=0,
                          scale=0.25,
                          flipud=0.5,
                          lr0=opt['lr0'],
                          lrf=opt['lrf'],
                          momentum=opt['momentum'],
                          weight_decay=opt['weight_decay'],
                          seed=123)
    wandb.finish()
