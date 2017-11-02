import json

path = 'E:/ML/SceneClassify/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'

with open(path) as f:
    a = json.load(f)
    print(len(a))
    print(a[0])
    image2label = {d['image_id']: d['label_id'] for d in a}
    print(image2label['79f993ae0858ae238b22968c5934d1ddba585ae4.jpg'])

