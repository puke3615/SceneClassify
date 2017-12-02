from classifier_base import BaseClassifier
from keras.applications import *
from keras.layers import *
from keras.engine import *
from config import *


class InceptionRestNetV2Classifier(BaseClassifier):
    def __init__(self, name='inception_resnet_v2', lr=1e-3, batch_size=BATCH_SIZE, weights_mode='loss', optimizer=None):
        BaseClassifier.__init__(self, name, IM_SIZE_299,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        weights = 'imagenet' if self.context['load_imagenet_weights'] else None
        model_inception_resnet_v2 = InceptionResNetV2(include_top=False, weights=weights,
                                                      input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        for layer in model_inception_resnet_v2.layers[:-50]:
            layer.trainable = False
        x = model_inception_resnet_v2.output
        x = Dense(CLASSES, activation='softmax')(x)
        model = Model(inputs=model_inception_resnet_v2.inputs, outputs=x)
        return model

    def data_generator(self, path_image, train=True, random_prob=1., **kwargs):
        return BaseClassifier.data_generator(self, path_image, train, random_prob, **kwargs)

if __name__ == '__main__':
    classifier = InceptionRestNetV2Classifier('inception_resnet_v2_50', lr=1e-5)
    classifier.train()
