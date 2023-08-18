from fastcore.script import *
from ultralytics import YOLO
import torch

@call_parse
def train_model(base_model:str, # Base model to use. Either yolov8n, yolov8s, yolov8m, yolov8l or yolov8x
                fold_to_use:int, # Which fold of the dataset to use as validation (1-5)
                data_path:str, # Path to data folder
                outdir:str='runs', # Where to save the model, default 'runs'
                epochs:int=50, # How many epochs to train 
                patience:int=10, # Epochs to wait for no observable improvement for early stopping
                batch:int=-1, # Batch size to use
                imgsz:int=640, # Image size to use
                optimizer:str='SGD', # Optimizer to use, one of [SGD, Adam, AdamW, RMSProp]
                cos_lr:bool=False, # Whether to use cosine annealing
                ):
    
    if base_model not in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
        print('Invalid base model, defaulting to yolov8s')
        base_model = 'yolov8s'

    if fold_to_use < 1 or fold_to_use > 5:
        print('Invalid fold, defaulting to 1')
        fold_to_use = 1

    if optimizer not in ['SGD', 'Adam', 'AdamW', 'RMSProp']:
        print('Invalid optimizer, defaulting to SGD')
        optimizer = 'SGD'

    optimizer_params = {
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

    opt = optimizer_params[optimizer]

    print(f'Starting to train with parameters {base_model} {optimizer} {fold_to_use}')
    
    # Replace model path and project folder with your own
    model = YOLO(f'/scratch/project_2007454/ship_detection/yolo_models/{base_model}.pt')
    model.model = torch.compile(model.model)
    results = model.train(data=f'{data_path}/yolo_fold{fold_to_use}.yaml',
                          epochs=epochs,
                          patience=patience,
                          imgsz=imgsz,
                          batch=batch,
                          project=f'/scratch/project_2007454/ship_detection/{outdir}/{base_model}_{optimizer}',
                          name=f'fold_{fold_to_use}',
                          cache='ram',
                          exist_ok=True,
                          optimizer=optimizer,
                          cos_lr=cos_lr,
                          lr0=opt['lr0'],
                          lrf=opt['lrf'],
                          momentum=opt['momentum'],
                          weight_decay=opt['weight_decay'],
                          device=0,
                          scale=0.25,
                          flipud=0.5)
