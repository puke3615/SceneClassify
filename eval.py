import json
import time
from PIL import Image
import numpy as np
import os

# PATH_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'
PATH_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'

PATH_IMAGE = os.path.join(PATH_BASE, 'scene_validation_images_20170908')
PATH_REF = os.path.join(PATH_BASE, 'scene_validation_annotations_20170908.json')
PATH_SUBMIT = 'eval_json/resnet.json'


def get_batch(generator, images, width, height):
    n_batch = len(images)
    result = np.zeros([n_batch, height, width, 3])
    for i, file in enumerate(images):
        img = Image.open(file)
        img = img.resize((width, height))
        x = np.asarray(img, np.float32)
        x = generator.random_transform(x)
        x = generator.standardize(x)
        result[i, :, :, :] = x
    return result


def dump_json(model, generator, width, height, save_path=PATH_SUBMIT, batch_size=16, top=3, stop=True, evaluate=True):
    print('Start dump json...')
    result = []
    images = [os.path.join(PATH_IMAGE, file) for file in os.listdir(PATH_IMAGE)]
    n_images = len(images)
    n_batch = n_images // batch_size
    n_last_batch = n_images % batch_size

    def predict_batch(start, end):
        inputs = get_batch(generator, images[start: end], width, height)
        predictions = model.predict(inputs)
        predictions = np.argsort(predictions)
        predictions = predictions[:, -top:][:, ::-1]
        image_ids = [os.path.basename(image) for image in images[start: end]]
        return [{'image_id': image_ids[i], 'label_id': predictions[i, :].tolist()} for i in range(end - start)]

    import sys
    for batch in range(n_batch):
        index = batch * batch_size
        sys.stdout.write('\rDumping %d/%d' % (index, n_images))
        batch_result = predict_batch(index, index + batch_size)
        result.extend(batch_result)
    if n_last_batch:
        index = n_batch * batch_size
        sys.stdout.write('\rDumping %d/%d' % (index, n_images))
        batch_result = predict_batch(index, index + n_last_batch)
        result.extend(batch_result)
    sys.stdout.write('\n')
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(save_path, 'w') as f:
        json.dump(result, f)
        print('Dump finished.')
    if evaluate:
        main()
    if stop:
        exit(0)


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


def main():
    if not os.path.exists(PATH_SUBMIT):
        raise Exception('Submit result "%s" not found. Call dump_json to dump result first.' % PATH_SUBMIT)
    result = {'error': [], 'warning': [], 'score': None}
    START_TIME = time.time()
    SUBMIT = {}
    REF = {}
    try:
        SUBMIT, REF = __load_data(PATH_SUBMIT, PATH_REF, result)
    except Exception as error:
        result['error'].append(str(error))
    try:
        result = __eval_result(SUBMIT, REF, result)
    except Exception as error:
        result['error'].append(str(error))
    print('Evaluation time of your result: %f s' % (time.time() - START_TIME))
    print(result)
    print('Score is %s' % result['score'])


if __name__ == '__main__':
    main()
