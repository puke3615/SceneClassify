from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from tensorboard import *

import utils
import os

PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'

# PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
# PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'classes')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'classes')

# PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
# PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'scene_validation_images_20170908')

IM_WIDTH = 299
IM_HEIGHT = 299
BATCH_SIZE = 32
CLASSES = len(os.listdir(PATH_TRAIN_IMAGES))
EPOCH = 100
LEARNING_RATE = 1e-3

PATH_WEIGHTS = 'params/xception.h5'
PATH_WEIGHTS_SAVED = 'params/xception.{epoch:02d-{val_loss:.2f}-{val_acc:.4f}.h5'
PATH_SUMMARY = 'log/xception'
DUMP_JSON = False


def build_generator(path_image, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        # channel_shift_range=wrap(10.),
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=None,
    )

    return image_generator.flow_from_directory(
        path_image,
        classes=['%02d' % i for i in range(CLASSES)],
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
    )


if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    steps_validate = utils.calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES)
    val_generator = build_generator(PATH_VAL_IMAGES, train=False)

    model_xception = Xception(include_top=False, weights='imagenet',
                              input_shape=(IM_HEIGHT, IM_WIDTH, 3), pooling='avg')
    for layer in model_xception.layers:
        layer.trainable = False
    x = model_xception.output
    x = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=model_xception.inputs, outputs=x)
    model.summary()

    adam = Nadam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    if os.path.exists(PATH_WEIGHTS):
        model.load_weights(PATH_WEIGHTS, True)
        print('Load %s successfully.' % PATH_WEIGHTS)
    else:
        print('Model params not found.')

    if DUMP_JSON:
        import eval

        eval.dump_json(model, val_generator.image_data_generator, IM_WIDTH, IM_HEIGHT)

    utils.ensure_dir(os.path.dirname(PATH_WEIGHTS))
    try:
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            callbacks=[
                ModelCheckpoint(PATH_WEIGHTS_SAVED, mode='max', save_best_only=True, verbose=1),
                StepTensorBoard(PATH_SUMMARY, skip_steps=200)
            ],
            epochs=EPOCH,
            validation_data=val_generator,
            validation_steps=steps_validate,
            verbose=1,
        )
    except KeyboardInterrupt:
        print('\nStop by keyboardInterrupt, try saving weights.')
        # model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
