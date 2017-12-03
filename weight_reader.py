from classifier_inception_resnet_v2 import *
from classifier_xception_trainable import *
from classifier_inception_v3 import *
from classifier_xception import *
from classifier_resnet import *
from classifier_vgg16 import *
from classifier_vgg19 import *
from predictor import *
from config import *
import numpy as np
import utils
import json
import eval
import os

"""
Input:  predictor
Output: class_weight, label_weight
"""


def dump_ndarray(data, file):
    try:
        dir = os.path.dirname(file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(file, data)
        print('Saved %s successfully.' % file)
    except Exception as e:
        raise Exception('Saved %s failure.' % file, e)


def load_ndarry(file):
    try:
        result = np.load(file)
        print('Load %s successfully.' % file)
        return result
    except:
        raise Exception('Load %s failure.' % file)


def create_weight_reader_by_predictor(predictor, batch_size=32, **predict_params):
    name = predictor.name
    func_predict = lambda files: predictor(files, top=1, **predict_params)
    weights = predictor.weights
    return WeightReader(name, func_predict, weights, batch_size=batch_size)


class WeightReader(object):
    def __init__(self, name, func_predict, weights, ref_file=PATH_VAL_JSON, batch_size=32):
        self.name = name
        self.func_predict = func_predict
        self.weights = weights
        self.ref_file = ref_file
        self.batch_size = batch_size

        self.context = CONTEXT(name)
        self.predictor_cache_dir = self.context['predictor_cache_dir']
        self.model_weight_path = self._get_weights_path('model_weight')
        self.label_weight_path = self._get_weights_path('label_weight')

    def _get_weights_path(self, sub_name):
        prefix = os.path.basename(self.weights).replace('.h5', '')
        filename = '%s_%s.npy' % (prefix, sub_name)
        return os.path.join(self.predictor_cache_dir, filename)

    def perform_eval(self):
        predictions, _ = eval.eval_predictor(self.func_predict, batch_size=self.batch_size)
        predictions = {image_id: label_ids[0] for image_id, label_ids in predictions}
        image2label = {}
        with open(self.ref_file, 'r') as f:
            ref_data = json.load(f)
            for item in ref_data:
                image2label[item['image_id']] = int(item['label_id'])
        assert len(predictions) == len(image2label), 'The predict length is not equal to ref length.'
        # image2label = {image: image2label[image] for image in predictions.keys()}
        right_count = 0
        n_labels = np.zeros((80,)) + 1e-5
        n_right_labels = np.zeros((80,))
        for image, label in image2label.items():
            prediction = predictions[image]
            n_labels[prediction] += 1
            # n_labels[label] += 1
            if prediction == label:
                right_count += 1
                n_right_labels[prediction] += 1
        model_weight = np.array([float(right_count) / max(len(image2label), 1e-5)])
        label_weight = n_right_labels / n_labels
        dump_ndarray(model_weight, self.model_weight_path)
        dump_ndarray(label_weight, self.label_weight_path)
        return model_weight, label_weight

    def get_model_weights(self, use_cache=True):
        if use_cache and os.path.isfile(self.model_weight_path):
            return load_ndarry(self.model_weight_path)
        return self.perform_eval()[0]

    def get_label_weights(self, use_cache=True):
        if use_cache and os.path.isfile(self.label_weight_path):
            return load_ndarry(self.label_weight_path)
        return self.perform_eval()[1]


if __name__ == '__main__':
    MODE = None
    predictors = [
        KerasPredictor(VGG19Classifier(), MODE),
        KerasPredictor(RestNetClassifier(), MODE),
        KerasPredictor(XceptionClassifier(), MODE),
        KerasPredictor(InceptionV3Classifier(), MODE),
        KerasPredictor(InceptionRestNetV2Classifier(), MODE),
    ]

    for predictor in predictors:
        print('\n[%s]' % predictor.name)
        reader = create_weight_reader_by_predictor(predictor, batch_size=128)
        model_weights = reader.get_model_weights(use_cache=True)
        label_weights = reader.get_label_weights()
        format_label_weights = label_weights + np.array(range(80))
        print(model_weights)
        print(format_label_weights)
