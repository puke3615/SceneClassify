from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from tensorboard import *
import keras

import utils
import os

# PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
# PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'

PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'classes10')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'classes10')

# PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
# PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'scene_validation_images_20170908')

IM_WIDTH = 224
IM_HEIGHT = 224
BATCH_SIZE = 32
CLASSES = len(os.listdir(PATH_TRAIN_IMAGES))
EPOCH = 100
LEARNING_RATE = 1e-2

PATH_WEIGHTS = 'params/class_10.h5'
PATH_SUMMARY = 'log/class_10'

def preprocess(x):
    noise = 10.
    v_min, v_max = np.min(x), np.max(x)
    noise = np.random.uniform(-noise, noise, x.shape).astype(np.float32)
    x = np.clip(x + noise, v_min, v_max)
    return x

def build_generator(path_image, train=True):
    def wrap(value):
        return float(train) and value

    image_generator = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=preprocess,
    )

    return image_generator.flow_from_directory(
        path_image,
        classes=['%02d' % i for i in range(CLASSES)],
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        save_to_dir='/Users/zijiao/Desktop/1',
    )

if __name__ == '__main__':
    file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
    steps_per_epoch = file_num // BATCH_SIZE
    steps_validate = utils.calculate_file_num(PATH_VAL_IMAGES) // BATCH_SIZE
    print('Steps number is %d every epoch.' % steps_per_epoch)
    train_generator = build_generator(PATH_TRAIN_IMAGES)
    val_generator = build_generator(PATH_VAL_IMAGES, train=False)

    model = Sequential()
    model.add(Conv2D(16, 3, activation='relu', padding='same', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    model.add(Dense(CLASSES, activation='softmax'))

    model.summary()

    adam = Nadam(lr=LEARNING_RATE)
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
            callbacks=[
                ModelCheckpoint(PATH_WEIGHTS),
                StepTensorBoard(PATH_SUMMARY, skip_steps=20)
            ],
            epochs=EPOCH,
            validation_data=val_generator,
            validation_steps=steps_validate,
        )
    except KeyboardInterrupt:
        print('\nStop by keyboardInterrupt, try saving weights.')
        model.save_weights(PATH_WEIGHTS)
        print('Save weights successfully.')
