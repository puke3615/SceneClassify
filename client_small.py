from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *

from generator import SenceDirectoryIterator
import utils
import os

# PATH_TRAIN_BASE = 'E:/ML/SceneClassify/ai_challenger_scene_train_20170904'
# PATH_VAL_BASE = 'E:/ML/SceneClassify/ai_challenger_scene_validation_20170908'
PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
PATH_TRAIN_JSON = os.path.join(PATH_TRAIN_BASE, 'scene_train_annotations_20170904.json')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'scene_validation_images_20170908')
PATH_VAL_JSON = os.path.join(PATH_VAL_BASE, 'scene_validation_annotations_20170908.json')
PATH_WEIGHTS = 'params/weights.h5'
IM_WIDTH = 128
IM_HEIGHT = 128
BATCH_SIZE = 128
CLASSES = 80
EPOCH = 50
LEARNING_RATE = 1e-2


def preprocess(x):
    x -= 127.5
    x /= 127.5
    return x


def build_generator(path_image, path_json, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        # rescale=1. / 255,
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.7),
        height_shift_range=wrap(0.7),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=preprocess,
    )

    return SenceDirectoryIterator(
        path_image,
        image_generator,
        path_json,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )


if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    steps_validate = utils.calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES, PATH_TRAIN_JSON)
    val_generator = build_generator(PATH_VAL_IMAGES, PATH_VAL_JSON, train=False)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(80, activation='softmax'))

    # optimizer = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if os.path.exists(PATH_WEIGHTS):
        model.load_weights(PATH_WEIGHTS, True)
        print('Load weights.h5 successfully.')
    else:
        print('Model params not found.')
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
