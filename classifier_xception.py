import utils
from classifier_base import BaseClassifier
from keras.applications import *
from keras.optimizers import *
from keras.layers import *
from keras.engine import *
from generator import *
from config import *


class XceptionClassifier(BaseClassifier):
    def __init__(self, name='xception', lr=2e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_299,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_xception = Xception(include_top=False, weights=weights,
                                  input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_xception.layers:
            layer.trainable = False
        x = model_xception.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_xception.inputs, outputs=x)
        return model

    def image_generator(self, train=True):
        def wrap(value):
            return float(train) and value

        return ImageDataGenerator(
            contrast_stretching=train,
            channel_shift_range=wrap(25.5),
            # rotation_range=wrap(15.),
            # width_shift_range=wrap(0.2),
            # height_shift_range=wrap(0.2),
            # shear_range=wrap(0.2),
            # zoom_range=wrap(0.2),
            horizontal_flip=train,
            preprocessing_function=scene_preprocess_input,
        )

    def data_generator(self, path_image, train=True):
        generator = BaseClassifier.data_generator(self, path_image, train)
        generator.crop_mode = None
        return generator


if __name__ == '__main__':
    # classifier = XceptionClassifier(lr=2e-3)
    # classifier = XceptionClassifier(lr=2e-4)
    # classifier = XceptionClassifier(lr=2e-5)
    classifier = XceptionClassifier('xception_resize', weights_mode='loss', optimizer=Adam(1e-4))
    classifier.train(class_weight=utils.calculate_class_weight())
