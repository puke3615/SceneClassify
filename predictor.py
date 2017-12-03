from classifier_base import BaseClassifier
from classifier_xception import XceptionClassifier
from classifier_vgg16 import VGG16Classifier
from im_utils import *
import weight_reader
from config import *
import numpy as np
import config
import utils
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


"""
A:  avg, 均值
B:  model_weight, 按模型acc加权
C:  label_weight, 按类别acc加权
D:  ada_boost, Adaboost方式
P:  prediction_weight, 按预测值本身加权
M:  Max, 
MM: Max with model weight
ML: Max with label weight
"""
POLICIES = ['A', 'B', 'C', 'D', 'P', 'M', 'MM', 'MP']


class IntegratedPredictor(object):
    def __init__(self, predictors, policies=POLICIES, weights=None, standard=False, all_combine=False,
                 use_weight_cache=True):
        self._check_policies(policies)
        self.predictors = predictors
        self.policies = policies
        self.weights = weights
        self.standard = standard
        self.all_combine = all_combine
        self.use_weight_cache = use_weight_cache

        self.model_weight = None
        self.label_weight = None
        self.index2combine_name = {}
        self.index2policy = {}
        self.combine_names = []

        self.names = [predictor.name for predictor in predictors]
        self.predictors_list = utils.all_combines(predictors) if all_combine else [predictors]
        self._parse_predictors_name()

    def _parse_predictors_name(self):
        index = 0
        for predictors in self.predictors_list:
            combine_name = self.get_name_by_predictors(predictors)
            self.combine_names.append(combine_name)
            if len(predictors) == 1:
                self.index2combine_name[index] = combine_name
                self.index2policy[index] = ''
                index += 1
            else:
                for policy in self.policies:
                    self.index2combine_name[index] = combine_name
                    self.index2policy[index] = policy
                    index += 1

    @staticmethod
    def get_name_by_predictors(predictors):
        return '[%s]' % ('#'.join([predictor.name for predictor in predictors]))

    def _check_policies(self, policies):
        for policy in policies:
            if policy not in POLICIES:
                raise Exception('Not support for "%s" policy. Choose from [%s].'
                                % (policy, ', '.join(POLICIES)))

    def __call__(self, files, top=3, return_with_prob=False, **kwargs):
        if isinstance(files, str):
            files = [files]
        # get prediction of every predictor
        predictions_dict = {predictor.name: predictor.perform_predict(files, **kwargs) for predictor in self.predictors}
        # integrated predictions
        final_predictions_dict = self.integrated_predictions(predictions_dict)
        # parse predictions
        top_predictions = []
        for combine_name in self.combine_names:
            item_predictions = final_predictions_dict[combine_name]
            top_predictions.extend([parse_prediction(files, item_prediction, top, return_with_prob)
                                    for item_prediction in item_predictions])
        return top_predictions

    def _parse_predictor(self, func_map):
        return {predictor.name: func_map(weight_reader.create_weight_reader_by_predictor(predictor))
                for predictor in self.predictors}

    def integrated_predictions(self, predictions_dict):
        integrated_predictions_dict = {}
        index = 0
        for predictors, combine_name in zip(self.predictors_list, self.combine_names):
            # single predictor, don't need to combine
            if len(predictors) == 1:
                integrated_predictions_dict[combine_name] = [predictions_dict[predictors[0].name]]
                self.index2combine_name[index] = combine_name
                self.index2policy[index] = ''
                index += 1
            else:
                predictions = [predictions_dict[predictor.name] for predictor in predictors]
                predictions = np.array(predictions)
                result = []
                for policy in self.policies:
                    result.append(self._perform_integrated(predictors, predictions, policy))
                    self.index2combine_name[index] = combine_name
                    self.index2policy[index] = policy
                    index += 1
                integrated_predictions_dict[combine_name] = result
        return integrated_predictions_dict

    def _perform_integrated(self, predictors, predictions, policy):
        if policy == 'A':
            result = np.mean(predictions, axis=0)
        elif policy == 'B':
            self._parse_mode_weight()
            assert self.model_weight, 'The weights is None.'
            c_ns = [self.model_weight[predictor.name] for predictor in predictors]
            assert len(c_ns) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(self.model_weight), len(predictions))
            result = np.sum(c_n * p_nj for c_n, p_nj in zip(c_ns, predictions))
        elif policy == 'C':
            self._parse_label_weight()
            c_njs = [self.label_weight[predictor.name] for predictor in predictors]
            result = np.sum(c_nj * p_nj for c_nj, p_nj in zip(c_njs, predictions))
        elif policy == 'D':
            self._parse_mode_weight()
            self._parse_label_weight()
            c_ns = [self.model_weight[predictor.name] for predictor in predictors]
            c_njs = [self.label_weight[predictor.name] for predictor in predictors]
            alphas = [np.log(c_n / (1 - c_n + 1e-6)) / 2 for c_n in c_ns]
            result = np.sum(c_nj * p_nj * alpha for alpha, c_nj, p_nj in zip(alphas, c_njs, predictions))
        elif policy == 'P':
            predictions_all = np.sum(predictions, axis=0)
            predictions_weight = predictions / predictions_all
            result = np.sum(predictions_weight * predictions, axis=0)
        elif policy == 'M':
            result = np.max(predictions, axis=0)
        elif policy == 'MM':
            self._parse_mode_weight()
            assert self.model_weight, 'The weights is None.'
            model_weight = [self.model_weight[predictor.name] for predictor in predictors]
            assert len(model_weight) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(self.model_weight), len(predictions))
            result = np.max([d * w for d, w in zip(predictions, model_weight)], axis=0)
        elif policy == 'ML':
            self._parse_label_weight()
            label_weight = [self.label_weight[predictor.name] for predictor in predictors]
            result = np.max([c_nj * p_nj for c_nj, p_nj in zip(label_weight, predictions)], axis=0)
        else:
            raise 'Not support for policy named "%s".' % policy
        if self.standard:
            denominators = np.sum(result, axis=1)
            result = [r / denominator for r, denominator in zip(result, denominators)]
        return result

    def _parse_label_weight(self):
        if not self.label_weight:
            self.label_weight = self._parse_predictor(
                lambda reader: reader.get_label_weights(use_cache=self.use_weight_cache))

    def _parse_mode_weight(self):
        if not self.model_weight:
            self.model_weight = self._parse_predictor(
                lambda reader: reader.get_model_weights(use_cache=self.use_weight_cache))


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
