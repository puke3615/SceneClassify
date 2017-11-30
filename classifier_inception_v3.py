from classifier_base import BaseClassifier
from keras.applications import *
from keras.layers import *
from keras.engine import *
from config import *


class InceptionV3Classifier(BaseClassifier):
    def __init__(self, name='inception_v3', lr=1e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_299,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_xception = InceptionV3(include_top=False, weights=weights,
                                     input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_xception.layers:
            layer.trainable = False
        x = model_xception.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_xception.inputs, outputs=x)
        return model


if __name__ == '__main__':
    classifier = InceptionV3Classifier(lr=1e-3)
    classifier.train()
