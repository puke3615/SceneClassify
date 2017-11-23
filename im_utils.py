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
        size = target_size
        w, h = img.size
        l, t, r, b = 0, 0, w, h
        offset = abs(w - h)
        if mode == 'train':
            # output 1 samples
            if w > h:
                l = random.randint(0, offset)
                r = l + h
            else:
                t = random.randint(0, offset)
                b = t + w
            img = img.crop((l, t, r, b)).resize([size, size], Image.BOX)
            outputs.append(np.asarray(img))
        elif mode == 'test':
            # output 6 samples
            images = []
            if w > h:
                # left
                images.append(img.crop((0, 0, h, h)))

                # center
                images.append(img.crop(((w - h) / 2, 0, (w + h) / 2, h)))

                # right
                images.append(img.crop((w - h, 0, w, h)))
            else:
                # top
                images.append(img.crop((0, 0, w, w)))

                # center
                images.append(img.crop(((h - w) / 2, 0, (h + w) / 2, w)))

                # bottom
                images.append(img.crop((h - w, 0, h, w)))
            for image in images:
                image = image.resize([size, size])
                outputs.append(np.asarray(image))
                outputs.append(np.asarray(image.transpose(Image.FLIP_LEFT_RIGHT)))
            patch = len(images)
    outputs = np.array(outputs)
    return outputs, patch
