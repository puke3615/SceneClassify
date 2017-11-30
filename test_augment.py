# coding=utf-8
from PIL import Image
import numpy as np
import time
import os

import im_utils
from config import *
from matplotlib import pyplot as plt
"""
测试图片增强效果
"""


COUNT = 4
COL = int(np.math.sqrt(COUNT))


def get_files(size, dir=os.path.join(PATH_TRAIN_IMAGES, '00')):
    return [os.path.join(dir, sub) for sub in np.random.choice(os.listdir(dir), size, replace=False)]


files = get_files(COUNT)
images = np.array(
    [np.asarray(Image.open(f).resize((299, 299))) for f in files],
    dtype=np.uint8
)

start = time.time()
for _ in range(1):
    # images = [im_utils.aug_images([image])[0] for image in images]
    images = im_utils.func_batch_handle_with_multi_process(images, True, standard=False)
print('Take %f seconds.' % (time.time() - start))

ROW = len(images) // COL + (0 if len(images) % COL == 0 else 1)
for r in range(ROW):
    for c in range(COL):
        i = r * COL + c
        if i >= len(images):
            break
        area = '%d%d%d' % (ROW, COL, c)
        plt.subplot(ROW, COL, i)
        plt.imshow(images[i], )
plt.show()
