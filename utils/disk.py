import os
import cv2
import yaml
import json
from typing import Any, Iterable

def get_paths_from_directory(directory: str, return_dict: bool = False) -> list[str] | dict[str, str]:
    if return_dict:
        return {x: os.path.join(directory, x) for x in os.listdir(directory)}
    
    return list(map(
        lambda x: os.path.join(directory, x), 
        os.listdir(directory)
    ))

def read_image(path: str, grayscale: bool = False) -> cv2.Mat:
    color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, color_mode)

def write_image(path: str, image: cv2.Mat) -> None:
    cv2.imwrite(path, image)
    

def write_yaml(path: str, data: dict) -> None:
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def read_yaml(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def write_json(path: str, data: dict) -> None:
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def read_json(path: str) -> dict:
    with open(path, 'r') as file:
        return json.load(file)

def write_jsonl(path: str, data: Any | Iterable[Any]) -> None:
    with open(path, 'a') as file:
        json.dump(data, file)
        file.write('\n')

def read_jsonl(path: str) -> list[dict]:
    with open(path, 'r') as file:
        data = list(map(json.loads, file))

    return data

def write_txt(path: str, data: str) -> None:
    with open(path, 'w') as file:
        file.write(data)

def read_txt(path: str) -> str:
    with open(path, 'r') as file:
        return file.read()

