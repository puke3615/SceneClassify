from classifier_base import BaseClassifier
from im_utils import *
import numpy as np


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
    def __init__(self, func_predict, target_size, mode='val', preprocess=default_preprocess_input,
                 top=3, return_with_prob=False):
        self.func_predict = func_predict
        self.target_size = target_size
        self.mode = mode
        self.top = top
        self.preprocess = preprocess
        self.return_with_prob = return_with_prob

    def __call__(self, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        predictions = self.perform_predict(files, **kwargs)
        return parse_prediction(files, predictions, self.top, self.return_with_prob)

    def perform_predict(self, files, **kwargs):
        inputs, patch = im2array(files, self.target_size, self.mode,
                                 preprocess=self.preprocess)
        assert patch * len(files) == len(inputs)
        predictions = self.func_predict(inputs, **kwargs)
        if patch != 1:
            predictions = np.array([np.mean(predictions[i: i + patch], axis=0) for i in range(0, len(inputs), patch)])
        return predictions


class GeneratorPreprocessConverter(object):
    def __init__(self, image_generator):
        self.image_generator = image_generator

    def __call__(self, x):
        x = self.image_generator.random_transform(x)
        x = self.image_generator.standardize(x)
        return x


class KerasPredictor(Predictor):
    def __init__(self, classifier, mode='val', preprocess=None,
                 top=3, return_with_prob=False):
        assert isinstance(classifier, BaseClassifier), \
            'The classifier is not a instance of %s' % (type(BaseClassifier))
        model = classifier.model
        if not preprocess:
            image_generator = classifier.image_generator(train=False)
            preprocess = GeneratorPreprocessConverter(image_generator)
        h, w = model.input_shape[1:3]
        assert h == w, 'Width is not equal with height.'
        Predictor.__init__(self, model.predict, w, mode,
                           preprocess, top, return_with_prob)

    def _check_classifier(self, func_or_files):
        fs = model.__class__.__dict__
        for f in func_or_files:
            if not fs.__contains__(f):
                raise Exception('No function or field named "%s" found in %s.' % (f, model.__class__.__name__))


class IntegratedPredictor(object):
    POLICIES = ['avg', 'weight', 'label_weight']
    DEFAULT_POLICY = 'avg'

    def __init__(self, predictors, policy='avg', weights=None, top=3, return_with_prob=False):
        self.predictors = predictors
        self.policy = policy if policy in self.POLICIES else self.DEFAULT_POLICY
        self.weights = weights
        self.top = top
        self.return_with_prob = return_with_prob

    def __call__(self, files, **kwargs):
        if isinstance(files, str):
            files = [files]
        predictions = [predictor.perform_predict(files, **kwargs) for predictor in self.predictors]
        if self.policy == 'avg':
            prediction_summary = np.mean(predictions, axis=0)
        elif self.policy == 'weight':
            assert self.weights, 'The weights is None.'
            assert len(self.weights) == len(predictions), \
                'The weights length %d is not equal with %d' % (len(self.weights), len(predictions))
            return np.sum(weight * prediction for weight, prediction in zip(self.weights, predictions))
        elif self.policy == 'label_weight':
            raise NotImplementedError('')
        else:
            raise Exception('No support for %s.' % self.policy)
        return parse_prediction(files, prediction_summary, self.top, self.return_with_prob)


if __name__ == '__main__':
    import classifier_xception

    path = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904/classes/00/0f39a480048e3e22640982a1c5fdde19bb5c2c47.jpg'
    model = classifier_xception.build_model()
    predictor = IntegratedPredictor([
        KerasPredictor(model, 'test'),
        KerasPredictor(model, 'val'),
    ], return_with_prob=False)
    prediction = predictor(path)
    print(prediction)
