from classifier_base import BaseClassifier
from classifier_xception import XceptionClassifier
from classifier_vgg16 import VGG16Classifier
from im_utils import *
import weight_reader
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
    def __init__(self, func_predict, target_size, mode=None, batch_handler=None):
        self.func_predict = func_predict
        self.target_size = target_size
        self.mode = mode
        self.batch_handler = batch_handler

    def __call__(self, files, top=3, return_with_prob=False, **kwargs):
        if isinstance(files, str):
            files = [files]
        predictions = self.perform_predict(files, **kwargs)
        return parse_prediction(files, predictions, top, return_with_prob)

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
    def __init__(self, classifier, mode=None, batch_handler=None):
        assert isinstance(classifier, BaseClassifier), \
            'The classifier is not a instance of %s' % (type(BaseClassifier))
        self.model = classifier.model
        self.weights = classifier.weights
        self.name = classifier.name
        # set default batch_handler if not exists
        if not batch_handler:
            batch_handler = lambda x: func_batch_handle_with_multi_process(x, False)
        h, w = self.model.input_shape[1:3]
        assert h == w, 'Width is not equal with height.'
        Predictor.__init__(self, self.model.predict, w, mode, batch_handler)


class IntegratedPredictor(object):
    POLICIES = ['avg', 'model_weight', 'label_weight', 'ada_boost']

    def __init__(self, predictors, policies='avg', weights=None, standard=False):
        self.predictors = predictors
        self.name = '[%s]' % ('#'.join([predictor.name for predictor in predictors]))
        if not isinstance(policies, list) and not isinstance(policies, tuple):
            policies = [policies]
            self._return_array = False
        else:
            self._return_array = True
        self.check_policies(policies)
        self.policies = policies
        self.weights = weights
        self.standard = standard
        self.model_weight = None
        self.label_weight = None

    def check_policies(self, policies):
        for policy in policies:
            if policy not in self.POLICIES:
                raise Exception('Not support for "%s" policy. Choose from [%s].'
                                % (policy, ', '.join(self.POLICIES)))

    def __call__(self, files, top=3, return_with_prob=False, **kwargs):
        if isinstance(files, str):
            files = [files]
        # get prediction of every predictor
        predictions = [predictor.perform_predict(files, **kwargs) for predictor in self.predictors]
        # integrated predictions
        final_predictions = self.integrated_predictions(predictions)
        # parse predictions
        top_predictions = [parse_prediction(files, item_prediction, top, return_with_prob)
                           for item_prediction in final_predictions]
        return top_predictions if self._return_array else top_predictions[0]

    def _parse_predictor(self, func_map):
        return [func_map(weight_reader.create_weight_reader_by_predictor(predictor))
                for predictor in self.predictors]

    def integrated_predictions(self, predictions):
        predictions = np.array(predictions)
        return [self._perform_integrated(predictions, policy) for policy in self.policies]

    def _perform_integrated(self, predictions, policy):
        if policy == 'avg':
            result = np.mean(predictions, axis=0)
        elif policy == 'model_weight':
            if not self.model_weight:
                self.model_weight = self._parse_predictor(lambda reader: reader.get_model_weights())
            assert self.model_weight, 'The weights is None.'
            assert len(self.model_weight) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(self.model_weight), len(predictions))
            c_ns = self.model_weight
            result = np.sum(c_n * p_nj for c_n, p_nj in zip(c_ns, predictions))
        elif policy == 'label_weight':
            if not self.label_weight:
                self.label_weight = self._parse_predictor(lambda reader: reader.get_label_weights())
            c_njs = self.label_weight
            result = np.sum(c_nj * p_nj for c_nj, p_nj in zip(c_njs, predictions))
        elif policy == 'ada_boost':
            if not self.model_weight:
                self.model_weight = self._parse_predictor(lambda reader: reader.get_model_weights())
            if not self.label_weight:
                self.label_weight = self._parse_predictor(lambda reader: reader.get_label_weights())
            c_ns = self.model_weight
            c_njs = self.label_weight
            alphas = [np.log(c_n / (1 - c_n + 1e-6)) / 2 for c_n in c_ns]
            result = np.sum(c_nj * p_nj * alpha for alpha, c_nj, p_nj in zip(alphas, c_njs, predictions))
        else:
            raise 'Not support for policy named "%s".' % policy
        if self.standard:
            denominators = np.sum(result, axis=1)
            result = [r / denominator for r, denominator in zip(result, denominators)]
        return result


if __name__ == '__main__':
    path = os.path.join(config.PATH_TRAIN_IMAGES, '00/919aa50cc17b08fa836eb3784349da0765131ab8.jpg')

    # single predictor
    predictor = KerasPredictor(VGG16Classifier(), 'val')

    # integrated predictor
    # predictor = IntegratedPredictor([
    #     KerasPredictor(classifier, 'test'),
    #     KerasPredictor(classifier, 'val'),
    # ], return_with_prob=True)

    prediction = predictor(path, return_with_prob=True)
    print(prediction)
