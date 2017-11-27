from generator import ImageDataGenerator
from config import *
import im_utils


def image_generator(train=True):
    def wrap(value):
        return float(train) and value

    return ImageDataGenerator(
        channel_shift_range=wrap(25.5),
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=im_utils.scene_preprocess_input,
    )


def data_generator(path_image, train=True):
    return image_generator(train).flow_from_directory(
        path_image,
        classes=['%02d' % i for i in range(80)],
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        crop_mode=None,
        save_prefix='train' if train else 'val',
        save_to_dir='/Users/zijiao/Desktop/1',
    )


generator = data_generator(PATH_TRAIN_IMAGES, train=True)
for i, (x, y) in enumerate(generator):
    if i >= 1:
        break
    print len(y)
