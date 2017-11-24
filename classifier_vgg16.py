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


if __name__ == '__main__':
    classifier = VGG16Classifier(lr=2e-3)
    classifier.train()
