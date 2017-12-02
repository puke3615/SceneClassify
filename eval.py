from classifier_inception_resnet_v2 import *
from classifier_inception_v3 import *
from classifier_xception import *
from classifier_resnet import *
from classifier_vgg16 import *
from classifier_vgg19 import *
from predictor import *
from PIL import Image
from config import *
import numpy as np
import im_utils
import utils
import json
import time
import os


# noinspection PyTypeChecker
def dump_json(predictor, save_path=PATH_JSON_DUMP, target_dir=PATH_VAL_IMAGES, batch_size=16):
    result = eval_predictor(predictor, target_dir, batch_size, dump_json_handler)
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(save_path, 'w') as f:
        json.dump(result, f)
        print('Dump finished.')


def dump_json_handler(image_id, label_id):
    return {'image_id': image_id, 'label_id': label_id}


def default_handler(image_id, label_id):
    return image_id, label_id


def eval_predictor(func_predict, target_dir=PATH_VAL_IMAGES,
                   batch_size=32, item_handler=default_handler):
    print('Start eval predictor...')
    result = []
    images = utils.get_files(target_dir)
    n_images = len(images)
    n_batch = n_images // batch_size
    n_last_batch = n_images % batch_size

    def predict_batch(start, end):
        predictions = func_predict(images[start: end])
        image_ids = [os.path.basename(image) for image in images[start: end]]
        return [item_handler(image_ids[i], predictions[i]) for i in range(end - start)]

    import sys
    for batch in range(n_batch):
        index = batch * batch_size
        batch_result = predict_batch(index, index + batch_size)
        result.extend(batch_result)
        sys.stdout.write('\rProcessing %d/%d' % (index + batch_size, n_images))
        sys.stdout.flush()
    if n_last_batch:
        index = n_batch * batch_size
        batch_result = predict_batch(index, index + n_last_batch)
        result.extend(batch_result)
        sys.stdout.write('\rProcessing %d/%d' % (index + n_last_batch, n_images))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return result


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

    result['score'] = str(float(right_count) / max(len(ref_dict), 1e-5))
    return result


def evaluate(eval_json, target_json):
    if not os.path.exists(eval_json):
        raise Exception('Submit result "%s" not found. Call dump_json to dump result first.' % PATH_JSON_DUMP)
    result = {'error': [], 'warning': [], 'score': None}
    START_TIME = time.time()
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
    print('Evaluation time of your result: %f s' % (time.time() - START_TIME))
    print(result)
    print('Score is %s' % result['score'])


DUMP_JSON = True
EVAL = True
MODE = None  # ['train', 'val', 'test', 'flip', None]
INTEGRATED_POLICY = 'avg'  # ['avg', 'model_weight', 'label_weight', 'ada_boost']
if __name__ == '__main__':
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
            ], policy=INTEGRATED_POLICY)

            dump_json(predictor, batch_size=128)
        finally:
            im_utils.recycle_pool()
    if EVAL:
        evaluate(PATH_JSON_DUMP, PATH_VAL_JSON)
