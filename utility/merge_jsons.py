import json
import os
import sys

# Get the path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Go up to EcoEyeSpy

# Allow user to specify input files or use defaults
if len(sys.argv) >= 3:
    training_path = sys.argv[1]
    testing_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, 'data.json')
else:
    # Use default paths - modify these as needed
    training_path = os.path.join(base_dir, 'training.json')
    testing_path = os.path.join(base_dir, 'testing.json')
    output_path = os.path.join(base_dir, 'data.json')

print(f"Merging JSON files:")
print(f"  Training: {training_path}")
print(f"  Testing: {testing_path}")
print(f"  Output: {output_path}")

with open(training_path) as training, open(testing_path) as testing:
    training_data = json.load(training)
    testing_data = json.load(testing)

merged = {'info': training_data['info'],
          'categories': training_data['categories'],
          'images': training_data['images'] + testing_data['images'],
          'annotations': testing_data['annotations']}

with open(output_path, 'w') as f:
    json.dump(merged, f, indent=4)
    
print(f"Successfully merged {len(merged['images'])} images")
print(f"Output saved to: {output_path}")