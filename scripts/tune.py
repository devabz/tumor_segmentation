import os
import optuna
from ultralytics import YOLO
from utils import disk
import numpy as np
from copy import deepcopy
from skimage.draw import polygon
import matplotlib.pyplot as plt
from utils import utils as cmp_utils
import gc


def load_data(data_dir):
    data_dir = os.path.join(os.getcwd(), data_dir)
    data = disk.get_paths_from_directory(data_dir, return_dict=True)
    data = {k:disk.get_paths_from_directory(v) for k, v in data.items()}
    data = {k:sorted(
        v, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0])
    ) for k, v in data.items()}
    return data


def parse_prediction(pred):
    mask = pred.masks
    if mask is None: return None

    (height, width) = mask.orig_shape
    response = np.zeros((height, width, 3), dtype=np.uint8)
    coords = deepcopy(mask.xyn)

    for coord in coords:
        coord[:, 0] *= width    # scale x-coordinates
        coord[:, 1] *= height   # scale y-coordinates

        row_index, col_index = polygon(
            coord[:, 1], 
            coord[:, 0], 
            (height, width)
        )
        
        response[row_index, col_index] = (255, 255, 255)
    
    return response


def evaluate_model(log_dir, conf=0.1, resolve_path=True):

    if not resolve_path: path = log_dir
    else: 
        path = os.path.join(os.getcwd(), log_dir)
        directories = disk.get_paths_from_directory(path)
        directories = sorted(
            directories,
            key=lambda x: int(x.split("/")[-1].replace('train', '0')),
        )
        path = directories[-1]
        
    model = YOLO(os.path.join(path, 'weights', 'best.pt'))

    output = model.predict(image_paths, conf=conf)
    output = list(map(parse_prediction, output))
    xyyhat = list(zip(images, labels, output))
    xyyhat = list(filter(lambda x: x[-1] is not None, xyyhat))

    try:
        passed = list(map(
            lambda x: cmp_utils.validate_segmentation(x[0], x[-1]), 
            xyyhat
        ))

    except Exception as e: return 0
    if not xyyhat: return 0

    del model
    gc.collect()

    return np.mean(
        list(map(
            lambda x: cmp_utils.dice_score(x[1], x[-1]),
            xyyhat
        ))
    )


def map_images_bw(image_paths):
    imgs = list(map(disk.read_image, image_paths))
    func = lambda x: np.dstack([x.max(-1, keepdims=True)]*3)
    return list(map(func, imgs))

def evaluate_model_bw(log_dir, conf=0.1, resolve_path=True):

    if not resolve_path: path = log_dir
    else: 
        path = os.path.join(os.getcwd(), log_dir)
        directories = disk.get_paths_from_directory(path)
        directories = sorted(
            directories,
            key=lambda x: int(x.split("/")[-1].replace('train', '0')),
        )
        path = directories[-1]
        
    model = YOLO(os.path.join(path, 'weights', 'best.pt'))
    imgs = map_images_bw(image_paths)
    output = model.predict(imgs, conf=conf)
    output = list(map(parse_prediction, output))
    xyyhat = list(zip(images, labels, output))
    xyyhat = list(filter(lambda x: x[-1] is not None, xyyhat))

    try:
        passed = list(map(
            lambda x: cmp_utils.validate_segmentation(x[0], x[-1]), 
            xyyhat
        ))

    except Exception as e: return 0
    if not xyyhat: return 0

    del model
    gc.collect()
    
    return np.mean(
        list(map(
            lambda x: cmp_utils.dice_score(x[1], x[-1]),
            xyyhat
        ))
    )



def train_model(kwargs):
    BASE_MODEL = 'yolo11n-seg'
    log_dir = f"solutions/task/tumor/artifacts/logs/{BASE_MODEL}/optuna/{STUDY_NAME}"
    model = YOLO(f"{BASE_MODEL}.pt")
    model.train(
        data="solutions/task/tumor/artifacts/configs/config.yaml",
        project=log_dir,
        epochs=EPOCHS, 
        imgsz=640, 
        lr0=0.005,
        lrf=0.005,
        optimizer="AdamW",
        patience=EPOCHS,
        **kwargs
    )

    return log_dir



def objective(trial):
    grid = {
        'cos_lr': trial.suggest_categorical('cos_lr', [True, False]),
        'momentum': trial.suggest_float('momentum', 0.9, 0.99, step=0.01),
        'weight_decay': trial.suggest_float('weight_decay', 0.00001, 0.005, step=0.00001),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 250, step=50),
        'warmup_momentum': trial.suggest_float('warmup_momentum', 0.7, 0.99, step=0.01),
        'warmup_bias_lr': trial.suggest_float('warmup_bias_lr', 0.001, 0.3, step=0.001),
    }
    
    log_dir = train_model(grid)
    score = evaluate_model(log_dir)
    return score


data = load_data('dm_i_ai_2025/tumor_segmentation/data/patients')
image_paths = data['imgs'][-10:]
images = list(map(disk.read_image, image_paths))
labels = list(map(disk.read_image, data['labels'][-10:]))

if __name__ == "__main__":
        
    EPOCHS = 500
        
    STUDY_NAME = "tune_yolo11n_seg_v0"
    study = optuna.create_study(
        storage="sqlite:///solutions/task/tumor/artifacts/db.sqlite3",
        study_name=STUDY_NAME,
        load_if_exists=True,
        direction='maximize',
    )

    study.optimize(objective, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})")