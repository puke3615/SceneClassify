from classifier_inception_resnet_v2 import *
from classifier_inception_v3 import *
from classifier_xception import *
from classifier_resnet import *
from classifier_vgg16 import *
from classifier_vgg19 import *
from predictor import *
from config import *
import numpy as np
import im_utils
import utils
import json
import time
import sys
import os


# noinspection PyTypeChecker
def dump_json(predictor, target_dir=PATH_VAL_IMAGES, batch_size=16):
    if isinstance(predictor, IntegratedPredictor):
        n_predictors = len(predictor.index2combine_name)
        path_json_dumps = [CONTEXT(predictor.index2combine_name[i], policy=predictor.index2policy[i])['path_json_dump']
                           for i in range(n_predictors)]
    else:
        path_json_dumps = [CONTEXT(predictor.name)['path_json_dump']]
    results, return_array = eval_predictor(predictor, target_dir, batch_size, dump_json_handler)
    assert len(results) == len(path_json_dumps), 'The result length is not equal with path_json_dumps\'s.'

    for result, save_path in zip(results, path_json_dumps):
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(save_path, 'w') as f:
            json.dump(result, f)
            print('Dump %s finished.' % save_path)
    return path_json_dumps if return_array else path_json_dumps[0]


def dump_json_handler(image_id, label_id):
    return {'image_id': image_id, 'label_id': label_id}


def default_handler(image_id, label_id):
    return image_id, label_id


class Flag:
    value = True


def eval_predictor(func_predict, target_dir=PATH_VAL_IMAGES,
                   batch_size=32, item_handler=default_handler):
    print('Start eval predictor...')
    results = []
    return_array = Flag()
    images = utils.get_files(target_dir)
    n_images = len(images)
    n_batch = n_images // batch_size
    n_last_batch = n_images % batch_size

    def predict_batch(start, end):
        predictions = func_predict(images[start: end])
        if not utils.is_multi_predictions(predictions):
            predictions = [predictions]
            return_array.value = False
        if len(results) == 0:
            for i in range(len(predictions)):
                results.append([])
        else:
            assert len(results) == len(predictions), 'The predictions length is not equal with last time\'s.'
        image_ids = [os.path.basename(image) for image in images[start: end]]
        for index, prediction in enumerate(predictions):
            results[index].extend([item_handler(image_ids[i], prediction[i]) for i in range(end - start)])
            sys.stdout.write('\rProcessing %d/%d' % (end, n_images))
            sys.stdout.flush()

    for batch in range(n_batch):
        index = batch * batch_size
        predict_batch(index, index + batch_size)
    if n_last_batch:
        index = n_batch * batch_size
        predict_batch(index, index + n_last_batch)
    sys.stdout.write('\n')
    return results if return_array.value else results[0], return_array.value


def __load_data(submit_file, reference_file, result):
    # load submit result and reference result

    with open(submit_file, 'r') as file1:
        submit_data = json.load(file1)
    with open(reference_file, 'r') as file1:
        ref_data = json.load(file1)
    if len(submit_data) != len(ref_data):
        result['warning'].append('Inconsistent number of images between submission and reference data \n')
    submit_dict = {}
    ref_dict = {}
    for item in submit_data:
        submit_dict[item['image_id']] = item['label_id']
    for item in ref_data:
        ref_dict[item['image_id']] = int(item['label_id'])
    return submit_dict, ref_dict


def __eval_result(submit_dict, ref_dict, result):
    # eval accuracy

    right_count = 0
    for (key, value) in ref_dict.items():

        if key not in set(submit_dict.keys()):
            result['warning'].append('lacking image %s in your submission file \n' % key)
            print('warnning: lacking image %s in your submission file' % key)
            continue

        if value in submit_dict[key][:3]:
            right_count += 1

    result['score'] = float(right_count) / max(len(ref_dict), 1e-5)
    return result


def evaluate(eval_json, target_json):
    if not os.path.exists(eval_json):
        raise Exception('Submit result "%s" not found. Call dump_json to dump result first.' % PATH_JSON_DUMP)
    result = {'error': [], 'warning': [], 'score': None}
    SUBMIT = {}
    REF = {}
    try:
        SUBMIT, REF = __load_data(eval_json, target_json, result)
    except Exception as error:
        result['error'].append(str(error))
    try:
        result = __eval_result(SUBMIT, REF, result)
    except Exception as error:
        result['error'].append(str(error))
    if result['warning'] or result['error']:
        print(result)
    print('Score is %s.' % result['score'])
    return result['score']


DUMP_JSON = True
EVAL = True
MODE = 'flip'  # ['train', 'val', 'test', 'flip', None]
INTEGRATED_POLICY = ['A', 'B', 'C', 'D', 'P', 'M', 'MM', 'MP']  # POLICIES = ['A', 'B', 'C', 'D', 'P', 'M', 'MM', 'MP']
if __name__ == '__main__':
    START_TIME = time.time()
    if DUMP_JSON:
        try:
            # single predictor
            # predictor = KerasPredictor(InceptionRestNetV2Classifier(), MODE)

            # integrated predictor
            predictor = IntegratedPredictor([
                KerasPredictor(VGG19Classifier(), MODE),
                KerasPredictor(RestNetClassifier(), MODE),
                KerasPredictor(XceptionClassifier(), MODE),
                KerasPredictor(InceptionV3Classifier(), MODE),
                KerasPredictor(InceptionRestNetV2Classifier(), MODE),
            ], policies=INTEGRATED_POLICY, all_combine=True)

            path_json_dumps = dump_json(predictor, batch_size=128)
            if not isinstance(path_json_dumps, list):
                path_json_dumps = [path_json_dumps]
        finally:
            im_utils.recycle_pool()
    else:
        root_path = os.path.dirname(os.path.dirname(CONTEXT('mock')['path_json_dump']))
        path_json_dumps = utils.get_files(root_path)
    if EVAL:
        scores = []
        model_names = []
        for json_path in path_json_dumps:
            filename = os.path.basename(json_path)
            policy = filename.replace('result_', '').replace('.json', '') if filename.__contains__('result_') else ''
            combine_name = os.path.basename(os.path.dirname(json_path))
            model_name = '%s%s' % (combine_name, ('_%s' % policy) if policy else '')
            print('\n%s' % model_name)
            scores.append(evaluate(json_path, PATH_VAL_JSON))
            model_names.append(model_name)

        sort_index = np.argsort(scores)[::-1]
        print('\n[Sorted by scores:]')
        for index in sort_index:
            print('%.16f, %s' % (scores[index], model_names[index]))

    time_str = utils.format_time(time.time() - START_TIME)
    print('\nEvaluation time of your result: %s.' % time_str)
