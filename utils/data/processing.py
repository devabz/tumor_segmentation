import numpy as np
import pandas as pd
from itertools import chain

def extract_bounding_coordinates_from_label_image(label_image: np.ndarray, true_pixel_value=(0, 0, 255)) -> set:
    # Perform a filtering operation
    coords = np.argwhere((label_image == true_pixel_value).sum(-1) == 3)
    coords = pd.DataFrame(coords, columns=['y', 'x'])
    split_ys = coords.y.where(coords.y.sort_values().diff() > 1).dropna().astype(int).tolist()
    split_xs = coords.x.where(coords.x.sort_values().diff() > 1).dropna().astype(int).tolist()

    # For each x coordinate find the min/max y
    coords = coords.groupby('x')
    coords = coords.y.agg(['min', 'max'])

    # unpack
    coords = list(map(
        lambda x: ((x[0], x[1]['min']), (x[0], x[1]['max'])),
        coords.to_dict(orient='index').items()
    ))

    # flatten and remove duplicates
    coords = set(chain(*coords))
    coords = sorted(coords, key=lambda x: x[0])

    bucket = []
    buckets = {}
    for (x, y) in coords:
        if x in split_xs or y in split_ys:
            buckets[len(buckets)] = bucket
            bucket = []
        bucket.append((x, y))

    # if bucket: buckets[len(buckets)] = bucket
    buckets = {k:v for k, v in buckets.items() if len(v) >= 3}  
    buckets = {i:v for i, v in enumerate(buckets.values())}
    return coords, buckets


from skimage import measure
def extract_bounding_coordinates_from_label_image(label_image: np.ndarray, true_pixel_value=(0, 0, 255), level: float = 0.5) -> set:
    mask = (label_image == true_pixel_value).all(axis=-1)
    contours = measure.find_contours(mask, level=level)
    return list(map(lambda x: x[:, ::-1].tolist(), contours))