import json
import os

# Get the path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Go up to EcoEyeSpy
segmentation_dir = os.path.join(base_dir, 'segmentation_datasets')

input_path = os.path.join(segmentation_dir, 'patches_labels.json')
output_path = os.path.join(segmentation_dir, 'patches_bbox.json')

with open(input_path, 'r') as f:
    patches = json.load(f)

bbox = []

for patch in patches['patches']:
    if patch['segmentation']!=[]:
        xs = patch['segmentation'][::2]
        ys = patch['segmentation'][1::2]
        x0  = min(xs)
        x1  = max(xs)
        y0  = min(ys)
        y1  = max(ys)
        width = x1 - x0
        height = y1 - y0
        patch['bbox'] = [x0, y0, width, height]
    else:
        patch['bbox'] = []
    
    del patch['segmentation']


with open(output_path, 'w') as f:
    json.dump(patches, f, indent=2)
    
print(f"Converted segmentations to bounding boxes")
print(f"Output saved to: {output_path}")