from keras.applications.xception import *
from generator_wrapper import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from tensorboard import *
from lr_monitor import *
from generator import *
from config import *

import im_utils
import utils
import os


class BaseClassifier(object):
    def __init__(self, name, im_size, lr=1e-3, batch_size=BATCH_SIZE, weights_mode='loss', optimizer=None):
        # receive params
        self.name = name
        self.im_size = im_size
        self.lr = lr
        self.batch_size = batch_size
        self.weights_mode = weights_mode
        self.weights = None
        self.optimizer = optimizer

        # parse context
        self.context = CONTEXT(self.name)
        self.path_summary = self.context['summary']
        self.path_weights = self.context['weights']

        # build model
        self.model = self.build_model()
        self._compiled = False

    def data_generator(self, path_image, train=True, random_prob=0.5, **kwargs):
        return DirectoryIterator(
            path_image, None,
            classes=['%02d' % i for i in range(CLASSES)],
            target_size=(self.im_size, self.im_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            batch_handler=lambda x: func_batch_handle_with_multi_process(x, train, random_prob),
            **kwargs
        )

    def build_model(self):
        if self.weights_mode not in [None, 'acc', 'loss']:
            raise Exception('Weights set error.')

        model = self.create_model()

        if self.weights_mode:
            self.weights = utils.get_best_weights(os.path.dirname(self.path_weights), self.weights_mode)
            if self.weights:
                model.load_weights(self.weights)
                print('Load %s successfully.' % self.weights)
            else:
                print('Model params not found.')
        return model

    def create_model(self):
        # load_imagenet_weights = self.context['load_imagenet_weights']
        # model_xception = Xception(include_top=False,
        #                           weights='imagenet' if load_imagenet_weights else None,
        #                           input_shape=(self.im_size, self.im_size, 3), pooling='avg')
        # for layer in model_xception.layers:
        #     layer.trainable = False
        # x = model_xception.output
        # x = Dense(CLASSES, activation='softmax')(x)
        # model = Model(inputs=model_xception.inputs, outputs=x)
        # return model
        raise NotImplementedError('Nothing...')

    def compile_mode(self, force=False):
        if not self._compiled or force:
            self._compiled = True
            if not self.optimizer:
                self.optimizer = Adam(self.lr)
            self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def train(self, **kwargs):
        # calculate files number
        file_num = utils.calculate_file_num(PATH_TRAIN_IMAGES)
        steps_train = file_num // self.batch_size
        print('Steps number is %d every epoch.' % steps_train)
        steps_val = utils.calculate_file_num(PATH_VAL_IMAGES) // self.batch_size

        # build data generator
        train_generator = self.data_generator(PATH_TRAIN_IMAGES)
        val_generator = self.data_generator(PATH_VAL_IMAGES, train=False)

        # compile model if not
        self.compile_mode()

        # start training
        utils.ensure_dir(os.path.dirname(self.path_weights))
        weights_info = parse_weigths(self.weights) if self.weights else None
        init_epoch = weights_info[0] if weights_info else 0
        print('Start training from %d epoch.' % init_epoch)
        init_step = init_epoch * steps_train
        try:
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=steps_train,
                callbacks=[
                    ModelCheckpoint(self.path_weights, verbose=1),
                    StepTensorBoard(self.path_summary, init_steps=init_step, skip_steps=200),
                    LRMonitor(step=10),
                ],
                initial_epoch=init_epoch,
                epochs=EPOCH,
                validation_data=val_generator,
                validation_steps=steps_val,
                verbose=1,
                class_weight=utils.calculate_class_weight(),
                **kwargs
            )
        except KeyboardInterrupt:
            print('\nStop by keyboardInterrupt, try saving weights.')
            # model.save_weights(PATH_WEIGHTS)
            print('Save weights successfully.')
        finally:
            im_utils.recycle_pool()


if __name__ == '__main__':
    classifier = BaseClassifier('base', IM_SIZE_299, lr=2e-4)
    classifier.train()
