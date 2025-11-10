import json

patches = json.load(open('/Users/omar/Downloads/segmentation_datasets/patches_labels.json'))

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


with open('/Users/omar/Downloads/segmentation_datasets/patches_bbox.json', 'w') as f:
    json.dump(patches, f, indent=2)