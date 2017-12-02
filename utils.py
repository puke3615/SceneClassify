from itertools import combinations

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import keras.backend as K
import tensorflow as tf
import numpy as np
import config
import os


def get_files(dir):
    import os
    if not os.path.exists(dir):
        return []
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_files(sub_path)
    return result


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def calculate_class_weight(train_path=config.PATH_TRAIN_IMAGES):
    if not os.path.isdir(train_path):
        raise Exception('Dir "%s" not exists.' % train_path)
    n_classes = [len(os.listdir(os.path.join(train_path, subdir))) for subdir in os.listdir(train_path)]
    n_all = sum(n_classes)
    return [num / float(n_all) for num in n_classes]


def get_best_weights(path_weights, mode='acc', postfix='.h5'):
    if not os.path.isdir(path_weights):
        return None
    sub_files = os.listdir(path_weights)
    if not sub_files:
        return None
    target = sub_files[0]
    sub_files_with_metric = list(filter(lambda f: f.endswith(postfix) and f.__contains__('-'), sub_files))
    if sub_files_with_metric:
        try:
            weights_value = [file.replace(postfix, '').split('-')[-2:] for file in sub_files_with_metric]
            key_filename = 'filename'
            kw = ['loss', 'acc']
            weights_info = []
            for filename, value in zip(sub_files_with_metric, weights_value):
                item = dict((k, float(v)) for k, v in zip(kw, value))
                item[key_filename] = filename
                weights_info.append(item)
            if mode not in kw:
                mode = 'acc'
            if mode == 'loss':
                weights_info = list(sorted(weights_info, key=lambda x: x['loss']))
            elif mode == 'acc':
                weights_info = list(sorted(weights_info, key=lambda x: x['acc'], reverse=True))
            target = weights_info[0][key_filename]
            print('The best weights is %s, sorted by %s.' % (target, mode))
        except:
            print('Parse best weights failure, choose first file %s.' % target)
    else:
        print('No weights with metric found, choose first file %s.' % target)
    return os.path.join(path_weights, target)


def is_multi_predictions(predictions):
    if isinstance(predictions, np.ndarray):
        return len(predictions.shape) == 3
    element = predictions[0][0]
    return isinstance(element, list) \
           or isinstance(element, tuple) \
           or isinstance(element, np.ndarray)


def all_combines(data):
    result = []
    for i in range(len(data)):
        combines = list(combinations(data, i + 1))
        result.extend(combines)
    return result


def preprocess_image(im, width, height, train=True):
    size = min(im.shape[:2])
    im = tf.constant(im)
    if train:
        im = tf.random_crop(im, (size, size, 3))
        im = tf.image.resize_images(im, (width, height))
    else:
        im = tf.image.resize_image_with_crop_or_pad(im, height, width)
    im = K.get_session().run(im)
    return preprocess_input(im)


def image_generator(train=True, preprocess=preprocess_input):
    def wrap(value):
        return float(train) and value

    return ImageDataGenerator(
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        channel_shift_range=wrap(25.5),
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=preprocess,
    )
