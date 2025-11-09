import cv2
import numpy as np
import json, os

def rasterise_polygon(segmentation, image_size, value=1):
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)
    if not segmentation:
        return mask
    if isinstance(segmentation[0], (int, float)):
        segmentation = [segmentation]

    for seg in segmentation:
        if len(seg) < 6: 
            continue
        pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], value)
    return mask


with open('/Users/omar/Downloads/segmentation_datasets/patches_labels.json') as f:
    data = json.load(f)

os.makedirs('/Users/omar/Downloads/segmentation_datasets/mask', exist_ok=True)

for ann in data['patches']:
    fname = ann['file_name']
    seg = ann['segmentation']
    cat = ann['category_id']

    mask = rasterise_polygon(seg, (256, 256), value=cat)
    cv2.imwrite(f"/Users/omar/Downloads/segmentation_datasets/mask/{fname}", mask)
