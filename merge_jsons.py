import json
with open('/Users/omar/Downloads/training.json') as training, open('/Users/omar/Downloads/testing.json') as testing:
    training_data = json.load(training)
    testing_data = json.load(testing)

merged = {'info': training_data['info'],
          'categories': training_data['categories'],
          'images': training_data['images'] + testing_data['images'],
          'annotations': testing_data['annotations']}

with open('/Users/omar/Downloads/data.json', 'w') as f:
    json.dump(merged, f, indent=4)