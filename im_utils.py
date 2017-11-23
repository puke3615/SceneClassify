from PIL import Image
import numpy as np
import random


def im2array(files, target_size, mode='test', grayscale=False):
    if not isinstance(files, list) and not isinstance(files, tuple):
        files = [files]
    outputs = []
    patch = 1
    for file in files:
        img = Image.open(file)
        if grayscale:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
        if mode == 'train':
            img = random_crop(img, target_size)
            outputs.append(np.asarray(img))
        elif mode == 'test':
            images = boundary_crop(img, target_size)
            for image in images:
                outputs.append(np.asarray(image))
            patch = len(images)
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
        return img.crop(((w - h) / 2, 0, (w + h) / 2, h)).resize([size, size])
    else:
        return img.crop(((h - w) / 2, 0, (h + w) / 2, w)).resize([size, size])


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
    img = img.crop((l, t, r, b)).resize([size, size], Image.BOX)
    return img
