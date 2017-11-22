from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
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

IM_WIDTH = 224
IM_HEIGHT = 224
BATCH_SIZE = 32
CLASSES = len(os.listdir(PATH_TRAIN_IMAGES))
EPOCH = 100
LEARNING_RATE = 1e-3

PATH_WEIGHTS = 'params/vgg16/{epoch:05d}-{val_loss:.4f}-{val_acc:.4f}.h5'
PATH_SUMMARY = 'log/vgg16'
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

    model_vgg16 = VGG16(include_top=False, weights='imagenet',
                        input_shape=(IM_HEIGHT, IM_WIDTH, 3), pooling='avg')
    for layer in model_vgg16.layers:
        layer.trainable = False
    x = model_vgg16.output
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=model_vgg16.inputs, outputs=x)
    model.summary()

    adam = Nadam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    weights = utils.get_best_weights(os.path.dirname(PATH_WEIGHTS))
    if weights:
        model.load_weights(weights, True)
        print('Load %s successfully.' % weights)
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
                ModelCheckpoint(PATH_WEIGHTS, verbose=1),
                StepTensorBoard(PATH_SUMMARY, skip_steps=200)
            ],
            epochs=EPOCH,
            validation_data=val_generator,
            validation_steps=steps_validate,
            verbose=2,
        )
    except KeyboardInterrupt:
        print('\nStop by keyboardInterrupt, try saving weights.')
        # model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
