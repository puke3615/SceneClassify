from classifier_base import BaseClassifier
from keras.applications import *
from keras.optimizers import *
from keras.layers import *
from keras.engine import *
from im_utils import *
from config import *
from utils import *
import generator


class XceptionTrainableClassifier(BaseClassifier):
    def __init__(self, name='xception_trainable', lr=2e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_299,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_xception = Xception(include_top=False, weights=weights,
                                  input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_xception.layers[:-10]:
            layer.trainable = False
        x = model_xception.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_xception.inputs, outputs=x)
        return model

    def data_generator(self, path_image, train=True, random_prob=1.0, **kwargs):
        params = {
            'multi_thread': True,
            'queue_size': 10,
            'threads': 5,
            'log': True,
        }
        params = dict(params, **kwargs)
        return BaseClassifier.data_generator(self, path_image, train, random_prob, **params)


if __name__ == '__main__':
    classifier = XceptionTrainableClassifier(
        name='xception_aug',
        weights_mode='loss',
        optimizer=Adam(1e-3)
    )
    classifier.train()
