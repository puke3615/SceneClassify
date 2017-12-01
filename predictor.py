from classifier_base import BaseClassifier
from classifier_xception import XceptionClassifier
from classifier_vgg16 import VGG16Classifier
from im_utils import *
from config import *
import numpy as np
import config
import os


# predictions = model.predict(inputs)

def parse_prediction(files, predictions, top=3, return_with_prob=False):
    result = np.argsort(predictions)
    result = result[:, -top:][:, ::-1]
    assert len(files) == len(result)
    if return_with_prob:
        return [[(j, predictions[i][j]) for j in r] for i, r in enumerate(result)]
    else:
        return list(map(lambda x: x.tolist(), result))


class Predictor:
    def __init__(self, func_predict, target_size, mode='val',
                 batch_handler=None, top=3, return_with_prob=False):
        self.func_predict = func_predict
        self.target_size = target_size
        self.mode = mode
        self.top = top
        self.batch_handler = batch_handler
        self.return_with_prob = return_with_prob

    def __call__(self, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        predictions = self.perform_predict(files, **kwargs)
        return parse_prediction(files, predictions, self.top, self.return_with_prob)

    def perform_predict(self, files, **kwargs):
        inputs, patch = im2array(files, self.target_size, self.mode)
        assert patch * len(files) == len(inputs)
        if self.batch_handler:
            inputs = self.batch_handler(inputs)
        predictions = self.func_predict(inputs, **kwargs)
        if patch != 1:
            predictions = np.array([np.mean(predictions[i: i + patch], axis=0) for i in range(0, len(inputs), patch)])
        return predictions


class KerasPredictor(Predictor):
    def __init__(self, classifier, mode='val', batch_handler=None,
                 top=3, return_with_prob=False):
        assert isinstance(classifier, BaseClassifier), \
            'The classifier is not a instance of %s' % (type(BaseClassifier))
        model = classifier.model
        # set default batch_handler if not exists
        if not batch_handler:
            batch_handler = lambda x: func_batch_handle_with_multi_process(x, False)
        h, w = model.input_shape[1:3]
        assert h == w, 'Width is not equal with height.'
        Predictor.__init__(self, model.predict, w, mode,
                           batch_handler, top, return_with_prob)


class IntegratedPredictor(object):
    POLICIES = ['avg', 'model_weight', 'label_weight', 'ada_boost']
    DEFAULT_POLICY = 'avg'

    def __init__(self, predictors, policy='avg', weights=None,
                 top=3, return_with_prob=False, standard=False,
                 model_weight=None, label_weight=None):
        self.predictors = predictors
        self.policy = policy if policy in self.POLICIES else self.DEFAULT_POLICY
        self.weights = weights
        self.top = top
        self.return_with_prob = return_with_prob
        self.standard = standard
        self.model_weight = model_weight
        self.label_weight = label_weight

    def __call__(self, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        # get prediction of every predictor
        predictions = [predictor.perform_predict(files, **kwargs) for predictor in self.predictors]
        # integrated predictions
        integrated_predictions = self.integrated_prob(predictions)
        # parse predictions
        return parse_prediction(files, integrated_predictions, self.top, self.return_with_prob)

    def integrated_prob(self, predictions):
        predictions = np.array(predictions)
        result = None
        if self.policy == 'avg':
            result = np.mean(predictions, axis=0)
        elif self.policy == 'model_weight':
            assert self.model_weight, 'The weights is None.'
            assert len(self.model_weight) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(self.model_weight), len(predictions))
            c_ns = self.model_weight
            result = np.sum(c_n * p_nj for c_n, p_nj in zip(c_ns, predictions))
        elif self.policy == 'label_weight':
            c_njs = self.label_weight
            result = np.sum(c_nj * p_nj for c_nj, p_nj in zip(c_njs, predictions))
        elif self.policy == 'ada_boost':
            c_ns = self.model_weight
            c_njs = self.label_weight
            alphas = [np.log(c_n / (1 - c_n + 1e-6)) / 2 for c_n in c_ns]
            result = np.sum(c_nj * p_nj * alpha for alpha, c_nj, p_nj in zip(alphas, c_njs, predictions))
        if result is None:
            raise 'Not support for policy named "%s".' % self.policy
        elif self.standard:
            denominators = np.sum(result, axis=1)
            result = [r / denominator for r, denominator in zip(result, denominators)]
        return result


if __name__ == '__main__':
    path = os.path.join(config.PATH_TRAIN_IMAGES, '00/919aa50cc17b08fa836eb3784349da0765131ab8.jpg')

    # single predictor
    predictor = KerasPredictor(VGG16Classifier(), 'val', return_with_prob=True)

    # integrated predictor
    # predictor = IntegratedPredictor([
    #     KerasPredictor(classifier, 'test'),
    #     KerasPredictor(classifier, 'val'),
    # ], return_with_prob=True)

    prediction = predictor(path)
    print(prediction)
