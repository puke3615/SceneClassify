import os
import platform


def is_mac():
    return os_name.startswith('darwin')


def is_windows():
    return os_name.startswith('windows')


def parse_weigths(weights):
    if not weights \
            or not weights.endswith('.h5') \
            or not weights.__contains__('/') \
            or not weights.__contains__('-'):
        return None
    try:
        weights_info = weights.split('/')[-1].replace('.h5', '').split('-')
        if len(weights_info) != 3:
            return None
        epoch = int(weights_info[0])
        val_loss = float(weights_info[1])
        val_acc = float(weights_info[2])
        return epoch, val_loss, val_acc
    except Exception as e:
        raise Exception('Parse weights failure: %s', str(e))

def CONTEXT(name, **kwargs):
    return {
        'weights': 'params/%s/{epoch:05d}-{val_loss:.4f}-{val_acc:.4f}.h5' % name,
        'summary': 'log/%s' % name,
        'load_imagenet_weights': is_windows(),
    }


# image path
os_name = platform.system().lower()
if is_windows():
    PATH_TRAIN_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_train_20170904'
    PATH_VAL_BASE = 'G:/Dataset/SceneClassify/ai_challenger_scene_validation_20170908'
elif is_mac():
    PATH_TRAIN_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904'
    PATH_VAL_BASE = '/Users/zijiao/Desktop/ai_challenger_scene_validation_20170908'
else:
    raise Exception('No images configured on %s' % os_name)

PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'classes')
PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'classes')

# PATH_TRAIN_IMAGES = os.path.join(PATH_TRAIN_BASE, 'scene_train_images_20170904')
# PATH_VAL_IMAGES = os.path.join(PATH_VAL_BASE, 'scene_validation_images_20170908')



# train info
IM_SIZE_299 = 299
IM_SIZE_224 = 224
BATCH_SIZE = 32
CLASSES = len(os.listdir(PATH_TRAIN_IMAGES))
EPOCH = 100

if __name__ == '__main__':
    print(PATH_TRAIN_IMAGES)
    print(CONTEXT('test').values())
