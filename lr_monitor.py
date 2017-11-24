# coding=utf-8
from keras.callbacks import Callback
import keras.backend as K


class LRMonitor(Callback):
    def __init__(self, step=1):
        super(LRMonitor, self).__init__()
        self.step = step

    def on_batch_end(self, batch, logs=None):
        if batch % self.step == 0:
            # 然并卵
            # print 'lr: %f' % self.model.optimizer.lr.eval(K.get_session())
            pass
