# coding=utf-8
import json
import csv
import os

# 源文件路径
PATH_BASE_DIR = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
# 保存文件路径
PATH_SAVE_DIR = os.path.join(PATH_BASE_DIR, 'classes')
# 是否按照分类名保存
SUB_DIR_WITH_NAME = False

PATH_IMAGES = os.path.join(PATH_BASE_DIR, 'scene_train_images_20170904')
PATH_JSON = os.path.join(PATH_BASE_DIR, 'scene_train_annotations_20170904.json')
PATH_CSV = os.path.join(PATH_BASE_DIR, 'scene_classes.csv')
PRINT = False


def output(obj):
    if PRINT:
        if isinstance(obj, list) or isinstance(obj, tuple):
            for i in obj:
                print(i)
        else:
            print(obj)


def parse_labels():
    with open(PATH_CSV) as f:
        return [line[1] for line in csv.reader(f)]


def parse_mapping():
    with open(PATH_JSON) as f:
        mapping = json.load(f)
        image2label = {item['image_id']: int(item['label_id']) for item in mapping}
        label2image = {}
        for image, label in image2label.items():
            if not label2image.__contains__(label):
                label2image[label] = []
            label2image[label].append(image)
        return image2label, label2image


if __name__ == '__main__':
    labels = parse_labels()
    output(labels[:5])

    image2label, label2image = parse_mapping()
    output(label2image[0][:5])

    for label, images in label2image.items():
        label_format = unicode(labels[label], 'utf-8') if SUB_DIR_WITH_NAME else ('%02d' % label)
        sub_dir = os.path.join(PATH_SAVE_DIR, label_format)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        for image in images:
            with open(os.path.join(PATH_IMAGES, image), 'r') as old:
                with open(os.path.join(sub_dir, image), 'w') as new:
                    new.write(old.read())
                    exit(0)
