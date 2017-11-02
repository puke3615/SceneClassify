from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras

from generator import SenceDirectoryIterator
import utils
import json
import os

PATH_TRAIN_BASE = 'E:/ML/SceneClassify/ai_challenger_scene_train_20170904'
PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
PATH_TRAIN_JSON = os.path.join(PATH_TRAIN_BASE, 'scene_train_annotations_20170904.json')
PATH_VAL_BASE = 'E:/ML/SceneClassify/ai_challenger_scene_validation_20170908'
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'ai_challenger_scene_validation_20170908')
PATH_WEIGHTS = 'params/weights.h5'
IM_WIDTH = 224
IM_HEIGHT = 224
BATCH_SIZE = 256
CLASSES = 80
EPOCH = 50

if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # train_generator = train_data_gen.flow_from_directory(
    #     PATH_TRAIN_IMAGES,
    #     classes=CLASSES,
    #     target_size=(IM_WIDTH, IM_HEIGHT),
    #     batch_size=BATCH_SIZE,
    #     class_mode='categorical')

    train_generator = SenceDirectoryIterator(
        PATH_TRAIN_IMAGES,
        image_generator,
        PATH_TRAIN_JSON,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                           pooling=None, classes=80)
    utils.ensure_dir(os.path.dirname(PATH_WEIGHTS))
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=[ModelCheckpoint(PATH_WEIGHTS)],
        epochs=EPOCH
    )
    model.save_weights(PATH_WEIGHTS)
