from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.models import *
from keras.layers import *
import keras

import utils
import os

PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'classes1')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'classes1')
PATH_WEIGHTS = 'params/weights.h5'
IM_WIDTH = 128
IM_HEIGHT = 128
BATCH_SIZE = 64
CLASSES = 10
EPOCH = 50
LEARNING_RATE = 1e-2


def build_generator(path_image, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=wrap(15.),
        samplewise_center=True,
        samplewise_std_normalization=True,
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=None,
    )

    return DirectoryIterator(
        path_image,
        image_generator,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    steps_validate = utils.calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES)
    val_generator = build_generator(PATH_VAL_IMAGES, train=False)

    model = keras.applications.vgg16.VGG16(include_top=True, weights=None,
                                           input_shape=(IM_HEIGHT, IM_WIDTH, 3), classes=CLASSES)
    # model_vgg = keras.applications.VGG16(include_top=False, weights=None, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
    # model = Sequential(model_vgg.layers)
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(4096, activation='relu', name='fc1'))
    # model.add(BatchNormalization())
    # model.add(Dense(4096, activation='relu', name='fc2'))
    # model.add(BatchNormalization())
    # model.add(Dense(CLASSES, activation='softmax'))
    model.summary()

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if os.path.exists(PATH_WEIGHTS):
        model.load_weights(PATH_WEIGHTS, True)
        print('Load weights.h5 successfully.')
    else:
        print('Model params not found.')

    utils.ensure_dir(os.path.dirname(PATH_WEIGHTS))
    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[ModelCheckpoint(PATH_WEIGHTS)],
            epochs=EPOCH,
            validation_data=val_generator,
            validation_steps=steps_validate,
        )
    except KeyboardInterrupt:
        print('Stop by keyboardInterrupt, try saving weights.')
        model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
