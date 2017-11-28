from generator import ImageDataGenerator
from config import *
import im_utils


def image_generator(train=True):
    def wrap(value):
        return float(train) and value

    return ImageDataGenerator(
        contrast_stretching=True,  #####
        histogram_equalization=False,  #####
        adaptive_equalization=False,  #####
        channel_shift_range=wrap(25.5),
        rotation_range=wrap(15.),
        width_shift_range=wrap(0.2),
        height_shift_range=wrap(0.2),
        shear_range=wrap(0.2),
        zoom_range=wrap(0.2),
        horizontal_flip=train,
        preprocessing_function=im_utils.scene_preprocess_input,
    )


def data_generator(path_image, train=True):
    return image_generator(train).flow_from_directory(
        path_image,
        classes=['%02d' % i for i in range(80)],
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        crop_mode=None,
        save_prefix='train' if train else 'val',
        save_to_dir='/Users/zijiao/Desktop/1',
    )


generator = data_generator(PATH_TRAIN_IMAGES, train=False)
for i, (x, y) in enumerate(generator):
    if i >= 1:
        break
<<<<<<< HEAD
    print len(y)

# from PIL import Image
# from skimage import exposure
# import numpy as np
# path = os.path.join(PATH_TRAIN_IMAGES, '00/0d8575935a771b6a64aa0bf769ae87453beefcbf.jpg')
# im = Image.open(path)
# # im.show()
#
# im = np.array(im)
# p2, p98 = np.percentile(im, (2, 98)) #####
# # im = exposure.rescale_intensity(im, in_range=(p2, p98)) #####
#
# # im = exposure.equalize_adapthist(im, clip_limit=0.03) #####
# # im *= 255
#
# # im = exposure.equalize_hist(im).astype(np.uint8) #####
# # im *= 255
#
# im = Image.fromarray(im.astype(np.uint8))
# im.show()
=======
    print(len(y))
>>>>>>> reduce adam lr for xception classifier
