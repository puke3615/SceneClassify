import json
import os

PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
path = os.path.join(PATH_TRAIN_BASE, 'scene_train_annotations_20170904.json')

with open(path) as f:
    a = json.load(f)
    print(len(a))
    print(a[0])
    image2label = {d['image_id']: int(d['label_id']) for d in a}
    print(type(image2label['79f993ae0858ae238b22968c5934d1ddba585ae4.jpg']))


a = False

print(float(a) and 0.1)
