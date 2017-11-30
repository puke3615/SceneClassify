from keras.models import Sequential
from classifier_base import BaseClassifier
from keras.layers import *
from config import *


class SmallClassifier(BaseClassifier):
    def __init__(self, name='small', lr=1e-3, batch_size=BATCH_SIZE, weights_mode='acc', optimizer=None):
        BaseClassifier.__init__(self,name, IM_SIZE_224,
                                lr, batch_size, weights_mode, optimizer)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, 3, activation='relu', padding='same',
                         input_shape=(self.im_size, self.im_size, 3)))
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
        return model


if __name__ == '__main__':
    classifier = SmallClassifier(lr=1e-3)
    classifier.train()
