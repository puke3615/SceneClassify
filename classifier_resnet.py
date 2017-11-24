from classifier_base import BaseClassifier
from keras.applications import *
from keras.layers import *
from keras.engine import *
from config import *


class RestNetClassifier(BaseClassifier):
    def __init__(self, lr=2e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self, 'resnet', IM_SIZE_224,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_resnet = ResNet50(include_top=False, weights=weights,
                                input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_resnet.layers:
            layer.trainable = False
        x = model_resnet.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_resnet.inputs, outputs=x)
        return model


if __name__ == '__main__':
    classifier = RestNetClassifier(lr=2e-3)
    classifier.train()
