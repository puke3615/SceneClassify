import utils
from classifier_base import BaseClassifier
from keras.applications import *
from keras.optimizers import *
from keras.layers import *
from keras.engine import *
from generator import *
from config import *


class XceptionClassifier(BaseClassifier):
    def __init__(self, name='xception', lr=1e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_299,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_xception = Xception(include_top=False, weights=weights,
                                  input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_xception.layers[:-100]:
            layer.trainable = False
        x = model_xception.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_xception.inputs, outputs=x)
        return model

    def data_generator(self, path_image, train=True, random_prob=1., **kwargs):
        BaseClassifier.data_generator(self, path_image, train, random_prob, **kwargs)

if __name__ == '__main__':
    classifier = XceptionClassifier(lr=1e-3, weights_mode='loss')
    classifier.train()
