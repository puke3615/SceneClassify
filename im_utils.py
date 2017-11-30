# coding=utf-8
from imgaug import augmenters as iaa
from PIL import Image
import imgaug as ia
import numpy as np
import random


class PoolHolder(object):
    def __init__(self, pool=None):
        self.pool = pool


holder = PoolHolder()


def recycle_pool():
    if holder.pool:
        holder.pool.close()


def _process_image_worker(tup):
    process, img, random_prob = tup
    ret = process(img, random_prob)
    return ret


def func_batch_handle_with_multi_process(batch_x, train=True, random_prob=0.5, standard=True):
    if train:
        if not holder.pool:
            import multiprocessing
            holder.pool = multiprocessing.Pool()
        result = holder.pool.map(
            _process_image_worker,
            ((aug_images_single, image, random_prob) for image in batch_x)
        )
        batch_x = np.array(result)
    if standard:
        batch_x = scene_preprocess_input(batch_x)
    return batch_x


def aug_images_single(images_data, random_prob):
    return aug_images([images_data], random_prob)[0]


def func_batch_handle(batch_x, train=True, random_prob=0.5):
    if train:
        batch_x = aug_images(batch_x, random_prob)
    batch_x = scene_preprocess_input(batch_x)
    return batch_x


def aug_images(images_data, random_prob=0.5):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(random_prob, aug)
    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            # iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-8, 8),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                       [
                           # Convert some images into their superpixel representation,
                           # sample between 20 and 200 superpixels per image, but do
                           # not replace all superpixels with their average, only
                           # some of them (p_replace).
                           # 超分辨率
                           sometimes(
                               iaa.Superpixels(
                                   p_replace=(0, 0.3),
                                   n_segments=(20, 200)
                               )
                           ),

                           # Blur each image with varying strength using
                           # gaussian blur (sigma between 0 and 3.0),
                           # average/uniform blur (kernel size between 2x2 and 7x7)
                           # median blur (kernel size between 3x3 and 11x11).
                           # 模糊
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 2.0)),
                               iaa.AverageBlur(k=(2, 4)),
                               iaa.MedianBlur(k=(3, 5)),
                           ]),

                           # Sharpen each image, overlay the result with the original
                           # image using an alpha between 0 (no sharpening) and 1
                           # (full sharpening effect).
                           # 锐化、亮度
                           iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),

                           # Same as sharpen, but for an embossing effect.
                           # 浮雕
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),

                           # Search in some images either for all edges or for
                           # directed edges. These edges are then marked in a black
                           # and white image and overlayed with the original image
                           # using an alpha of 0 to 0.7.
                           # 边缘检测
                           sometimes(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0, 0.3)),
                               iaa.DirectedEdgeDetect(
                                   alpha=(0, 0.7), direction=(0.0, 1.0)
                               ),
                           ])),

                           # Add gaussian noise to some images.
                           # In 50% of these cases, the noise is randomly sampled per
                           # channel and pixel.
                           # In the other 50% of all cases it is sampled once per
                           # pixel (i.e. brightness change).
                           # 高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # Either drop randomly 1 to 10% of all pixels (i.e. set
                           # them to black) or drop them on an image with 2-5% percent
                           # of the original size, leading to large dropped
                           # rectangles.
                           # 点缀
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),

                           # Invert each image's chanell with 5% probability.
                           # This sets each pixel value v to 255-v.
                           # 通道逆反
                           iaa.Invert(0.05, per_channel=True),  # invert color channels

                           # Add a value of -10 to 10 to each pixel.
                           # 颜色通道偏移
                           iaa.Add((-10, 10), per_channel=0.5),

                           # Change brightness of images (50-150% of original value).
                           # 亮度
                           iaa.Multiply((0.7, 1.3), per_channel=0.5),

                           # Improve or worsen the contrast of images.
                           # 对比度
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                           # Convert each image to grayscale and then overlay the
                           # result with the original with random alpha. I.e. remove
                           # colors with varying strengths.
                           # 灰度
                           iaa.Grayscale(alpha=(0.0, 1.0)),

                           # In some images move pixels locally around (with random
                           # strengths).
                           # 弹性转换
                           sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)
                           ),

                           # In some images distort local areas with varying strength.
                           # 分段仿射变换
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03)))
                       ],
                       # do all of the above augmentations in random order
                       random_order=True
                       )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )
    return seq.augment_images(images_data)


def im2array(files, target_size, mode=None, preprocess=None):
    def handle(im):
        im.flags.writeable = True
        im = im.astype(np.float32)
        if callable(preprocess):
            im = preprocess(im)
        return im

    if not isinstance(files, list) and not isinstance(files, tuple):
        files = [files]
    if mode not in ['train', 'val', 'test', None]:
        raise Exception('The mode named "%s" not define.' % mode)
    outputs = []
    for file in files:
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if mode == 'train':
            img = random_crop(img, target_size)
            outputs.append(handle(np.asarray(img)))
        elif mode == 'val':
            img = center_crop(img, target_size)
            outputs.append(handle(np.asarray(img)))
        elif mode == 'test':
            images = boundary_crop(img, target_size)
            for image in images:
                outputs.append(handle(np.asarray(image)))
        elif mode is None:
            img = img.resize((target_size, target_size))
            outputs.append(handle(np.asarray(img)))
    patch = len(outputs) // len(files)
    outputs = np.array(outputs)
    return outputs, patch


def boundary_crop(img, target_size):
    # output 6 samples
    size = target_size
    w, h = img.size
    images = []
    if w > h:
        # left
        images.append(img.crop((0, 0, h, h)).resize([size, size]))

        # center
        images.append(img.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize([size, size]))

        # right
        images.append(img.crop((w - h, 0, w, h)).resize([size, size]))
    else:
        # top
        images.append(img.crop((0, 0, w, w)).resize([size, size]))

        # center
        images.append(img.crop(((h - w) // 2, 0, (h + w) // 2, w)).resize([size, size]))

        # bottom
        images.append(img.crop((h - w, 0, h, w)).resize([size, size]))
    images += [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
    return images


def center_crop(img, target_size):
    # output 6 samples
    size = target_size
    w, h = img.size
    if w > h:
        return img.crop(((w - h) // 2, 0, (w + h) // 2, h)).resize([size, size])
    else:
        return img.crop((0, (h - w) // 2, w, (h + w) // 2)).resize([size, size])


def random_crop(img, target_size):
    # output 1 samples
    size = target_size
    w, h = img.size
    l, t, r, b = 0, 0, w, h
    offset = abs(w - h)
    if w > h:
        l = random.randint(0, offset)
        r = l + h
    else:
        t = random.randint(0, offset)
        b = t + w
    img = img.crop((l, t, r, b)).resize([size, size])
    return img


def scene_preprocess_input(x):
    # Mean is [0.4960301824223457, 0.47806493084428053, 0.44767167301470545]
    # Var is [0.084966025569294362, 0.082005493489533315, 0.088877477602068156]
    if x.dtype == np.uint8:
        x = x.astype(np.float32)
    scale = 1 / 255.
    x *= scale
    if len(x.shape) == 3:
        # mean
        x[:, :, 0] -= 0.4960301824223457
        x[:, :, 1] -= 0.47806493084428053
        x[:, :, 2] -= 0.44767167301470545
        # var
        x[:, :, 0] /= 0.084966025569294362
        x[:, :, 1] /= 0.082005493489533315
        x[:, :, 2] /= 0.088877477602068156
    elif len(x.shape) == 4:
        # mean
        x[:, :, :, 0] -= 0.4960301824223457
        x[:, :, :, 1] -= 0.47806493084428053
        x[:, :, :, 2] -= 0.44767167301470545
        # var
        x[:, :, :, 0] /= 0.084966025569294362
        x[:, :, :, 1] /= 0.082005493489533315
        x[:, :, :, 2] /= 0.088877477602068156
    else:
        raise Exception('Format error.')
    return x


def scene_preprocess_input_with_resize_299x299(x):
    # Mean is [0.49444094156654222, 0.47744633346506349, 0.44751775014165357]
    # Var is [0.084434645233592204, 0.081736530937098234, 0.088808324105198816]
    scale = 1 / 255.
    x *= scale
    # mean
    x[:, :, 0] -= 0.49444094156654222
    x[:, :, 1] -= 0.47744633346506349
    x[:, :, 2] -= 0.44751775014165357
    # var
    x[:, :, 0] /= 0.084434645233592204
    x[:, :, 1] /= 0.081736530937098234
    x[:, :, 2] /= 0.088808324105198816
    return x


def default_preprocess_input(x):
    x = np.divide(x, 255.)
    x -= 0.5
    x *= 2.
    return x


def preprocess_input(x, rescale=1 / 255., center=True, normalization=True):
    if rescale:
        x *= rescale
    if center:
        x = center_handle(x)
    if normalization:
        x = std_normalization(x)
    return x


def std_normalization(x):
    x /= np.std(x, keepdims=True) + 1e-7
    return x


def center_handle(x):
    x -= np.mean(x, keepdims=True)
    return x


if __name__ == '__main__':
    path = '/Users/zijiao/Desktop/fzq.jpeg'
    im = Image.open(path)
    a = center_crop(im, 299)
    a.show()
