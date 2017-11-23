import keras.applications
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *

from generator import SenceDirectoryIterator
import utils
import os

PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'
# PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
# PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
# PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'classes')
PATH_TRAIN_JSON = os.path.join(PATH_TRAIN_BASE, 'scene_train_annotations_20170904.json')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'scene_validation_images_20170908')
PATH_VAL_JSON = os.path.join(PATH_VAL_BASE, 'scene_validation_annotations_20170908.json')
PATH_WEIGHTS = 'params/weights.h5'
IM_WIDTH = 128
IM_HEIGHT = 128
BATCH_SIZE = 32
CLASSES = 80
EPOCH = 50
LEARNING_RATE = 1e-2


def preprocess(x):
    x = np.subtract(x, 127.5)
    x = np.divide(x, 127.5)
    return x


def build_generator(path_image, path_json, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.7),
        height_shift_range=wrap(0.7),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=None,
    )

    return SenceDirectoryIterator(
        path_image,
        image_generator,
        path_json,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )


def build_model(weights_mode='acc', compile=False):
    model_vgg = keras.applications.VGG16(include_top=False, weights=None, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
    model = Sequential(model_vgg.layers)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    model.add(Dense(CLASSES, activation='softmax'))
    if compile:
        optimizer = Adam(lr=LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if weights_mode not in [None, 'acc', 'loss']:
        raise Exception('Weights set error.')
    if weights_mode:
        weights = utils.get_best_weights(os.path.dirname(PATH_WEIGHTS), weights_mode)
        if weights:
            model.load_weights(PATH_WEIGHTS, True)
            print('Load weights.h5 successfully.')
        else:
            print('Model params not found.')
    return model


if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    steps_validate = utils.calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES, PATH_TRAIN_JSON)
    val_generator = build_generator(PATH_VAL_IMAGES, PATH_VAL_JSON, train=False)

    model = build_model(compile=True)
    try:
        utils.ensure_dir(os.path.dirname(PATH_WEIGHTS))
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[ModelCheckpoint(PATH_WEIGHTS)],
            epochs=EPOCH,
            validation_data=val_generator,
            validation_steps=steps_validate,
        )
    except KeyboardInterrupt:
        print('\nStop by keyboardInterrupt, try saving weights.')
        model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
