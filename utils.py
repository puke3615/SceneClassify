import os


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_best_weights(path_weights, mode='acc', postfix='.h5'):
    if not os.path.isdir(path_weights):
        return None
    sub_files = os.listdir(path_weights)
    if not sub_files:
        return None
    target = sub_files[0]
    sub_files_with_metric = list(filter(lambda f: f.endswith(postfix) and f.__contains__('-'), sub_files))
    if sub_files_with_metric:
        try:
            weights_value = [file.replace(postfix, '').split('-')[-2:] for file in sub_files_with_metric]
            key_filename = 'filename'
            kw = ['loss', 'acc']
            weights_info = []
            for filename, value in zip(sub_files_with_metric, weights_value):
                item = dict((k, float(v)) for k, v in zip(kw, value))
                item[key_filename] = filename
                weights_info.append(item)
            if mode not in kw:
                mode = 'acc'
            if mode == 'loss':
                weights_info = list(sorted(weights_info, key=lambda x: x['loss']))
            elif mode == 'acc':
                weights_info = list(sorted(weights_info, key=lambda x: x['acc'], reverse=True))
            target = weights_info[0][key_filename]
            print('The best weights is %s, sorted by %s.' % (target, mode))
        except:
            print('Parse best weights failure, choose first file %s.' % target)
    else:
        print('No weights with metric found, choose first file %s.' % target)
    return os.path.join(path_weights, target)


def preprocess_image(im, width, height, train=True):
    from keras.applications.xception import preprocess_input
    import keras.backend as K
    import tensorflow as tf
    size = min(im.shape[:2])
    im = tf.constant(im)
    if train:
        im = tf.random_crop(im, (size, size, 3))
        im = tf.image.resize_images(im, (width, height))
    else:
        im = tf.image.resize_image_with_crop_or_pad(im, height, width)
    im = K.get_session().run(im)
    return preprocess_input(im)
