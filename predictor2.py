from im_utils import *
from PIL import Image
import numpy as np
import random
import os


# predictions = model.predict(inputs)



class AvgPrediction(object):
    def __init__(self):
        pass

    def __call__(self, inputs, **kwargs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = np.array(inputs)
        return np.mean(inputs, axis=0)


class Predictor:
    def __init__(self, func_predict, target_size, mode='test', top=3):
        self.func_predict = func_predict
        self.file_reader = im2array(target_size, mode)
        self.target_size = target_size
        self.top = top

    def __call__(self, files, **kwargs):
        try:
            self.files = [files] if isinstance(files, str) else files
            array = self.to_array(files)
            predictions = self.to_predict(array)
            return self.to_result(predictions)
        finally:
            self.files = []

    def to_array(self, files):
        return self.file_reader(files)

    def to_predict(self, array):
        return self.func_predict(array)

    def to_result(self, predictions):
        predictions = np.argsort(predictions)
        predictions = predictions[:, -self.top:][:, ::-1]
        image_ids = [os.path.basename(file) for file in self.files]
        return [{'image_id': image_ids[i], 'label_id': predictions[i, :].tolist()} for _ in self.files]


class A:
    def __init__(self, value=1):
        self.value = value

    def predict(self, x, batch_size=32, verbose=0):
        return np.ones(x.shape) * self.value


def model_to_predictor(model):
    funcs = model.__class__.__dict__
    func_predict = 'predict'
    if not funcs.__contains__(func_predict):
        raise Exception('No function named "%s" found in model.' % func_predict)
    return Predictor(model.predict)


class Predictor:
    def __init__(self, func_predict):
        self.func_predict = func_predict

    def __call__(self, inputs, **kwargs):
        return self.func(self.model, kwargs)

    def _check_method(self):
        pass


class IntegratedPredictor(object):
    POLICIES = ['avg', 'weight', 'label_weight']
    DEFAULT_POLICY = 'avg'

    def __init__(self, models=None, policy='avg'):
        self.models = models
        self.policy = policy if policy in self.POLICIES else self.DEFAULT_POLICY
        self.predictors = [Predictor(model) for model in models]

    def __call__(self, *args, **kwargs):
        if self.policy == 'avg':
            predictions = [predictor(*args, **kwargs) for predictor in self.predictors]
            prediction_summary = np.mean(predictions, axis=0)
        elif self.policy == 'weight':
            raise NotImplementedError('')
        elif self.policy == 'label_weight':
            raise NotImplementedError('')
        else:
            raise Exception('No support.')
        return prediction_summary


if __name__ == '__main__':
    outputs, patch = im2array('/Users/zijiao/Desktop/test.jpg', 505, 'train')
    for output in outputs:
        Image.fromarray(output).show()
    print outputs.shape

    # print AvgPrediction()([np.ones((3, 4)) * i for i in range(1, 10)])
