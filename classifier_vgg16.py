import keras.optimizers

import generator
from classifier_base import BaseClassifier
from keras.applications import *
from keras.layers import *
from keras.engine import *
from config import *


class VGG16Classifier(BaseClassifier):
    def __init__(self, name='vgg16', lr=2e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_224,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_vgg16 = VGG16(include_top=False, weights=weights,
                            input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_vgg16.layers:
            layer.trainable = False
        x = model_vgg16.output
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_vgg16.inputs, outputs=x)
        return model

    def image_generator(self, train=True):
        def wrap(value):
            return float(train) and value

        return generator.ImageDataGenerator(
            contrast_stretching=train,
            channel_shift_range=wrap(25.5),
            rotation_range=wrap(6.),
            width_shift_range=wrap(0.05),
            height_shift_range=wrap(0.05),
            shear_range=wrap(0.05),
            zoom_range=wrap(0.05),
            horizontal_flip=train,
            fill_mode='constant',
            preprocessing_function=generator.scene_preprocess_input,
        )

    def data_generator(self, path_image, train=True):
        generator = BaseClassifier.data_generator(self, path_image, train)
        generator.crop_mode = None
        return generator


if __name__ == '__main__':
    classifier = VGG16Classifier('vgg16_little', optimizer=keras.optimizers.Adam())
    classifier.train()
