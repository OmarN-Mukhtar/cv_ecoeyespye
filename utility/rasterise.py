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


# Get the path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Go up to EcoEyeSpy
segmentation_dir = os.path.join(base_dir, 'segmentation_datasets')

input_path = os.path.join(segmentation_dir, 'patches_labels.json')
output_dir = os.path.join(segmentation_dir, 'mask')

print(f"Reading patches from: {input_path}")
print(f"Output masks to: {output_dir}")

with open(input_path) as f:
    data = json.load(f)

os.makedirs(output_dir, exist_ok=True)

for ann in data['patches']:
    fname = ann['file_name']
    seg = ann['segmentation']
    cat = ann['category_id']

    mask = rasterise_polygon(seg, (256, 256), value=cat)
    output_path = os.path.join(output_dir, fname)
    cv2.imwrite(output_path, mask)

print(f"Rasterised {len(data['patches'])} masks successfully")
