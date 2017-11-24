from PIL import Image
import numpy as np
import random


def im2array(files, target_size, mode=None, preprocess=None):
    def handle(im):
        im.flags.writeable = True
        im = im.astype(np.float32)
        if callable(preprocess):
            im = preprocess(im)
        return im

    if not isinstance(files, list) and not isinstance(files, tuple):
        files = [files]
    if mode not in ['train', 'val', 'test']:
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
            img = img.resize(target_size)
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
        images.append(img.crop(((w - h) / 2, 0, (w + h) / 2, h)).resize([size, size]))

        # right
        images.append(img.crop((w - h, 0, w, h)).resize([size, size]))
    else:
        # top
        images.append(img.crop((0, 0, w, w)).resize([size, size]))

        # center
        images.append(img.crop(((h - w) / 2, 0, (h + w) / 2, w)).resize([size, size]))

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
    scale = 1 / 255.
    x *= scale
    # mean
    x[:, :, 0] -= 0.4960301824223457
    x[:, :, 1] -= 0.47806493084428053
    x[:, :, 2] -= 0.44767167301470545
    # var
    x[:, :, 0] -= 0.084966025569294362
    x[:, :, 1] -= 0.082005493489533315
    x[:, :, 2] -= 0.088877477602068156
    return x


def default_preprocess_input(x):
    return preprocess_input(x, rescale=1 / 255., center=True, normalization=True)


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
