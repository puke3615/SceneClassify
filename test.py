# coding=utf-8
import numpy as np
import heapq

"""
一般性测试
"""

a = [
    [2, 1, 3],
    [4, 9, 5],
]
a = np.array(a)
b = a.argsort()[:, -2:][:, ::-1]
print(b)

path = 'params/xception.{epoch:05d}-{val_loss:.2f}-{val_acc:.4f}.h5'
epoch = 10
logs = {'val_loss': 2.7863322, 'val_acc': 0.654543}
print(path.format(epoch=epoch, **logs))

# def preprocess(x):
#     noise = 10.
#     v_min, v_max = np.min(x), np.max(x)
#     noise = np.random.uniform(-noise, noise, x.shape).astype(np.float32)
#     x = np.clip(x + noise, v_min, v_max)
#     return x
#
# from keras.preprocessing.image import random_channel_shift
# from PIL import Image
# path = '/Users/zijiao/Desktop/ai_challenger_scene_train_20170904/classes10/00/0c54acfe60493b1167186e456620a124813051a9.jpg'
# im = Image.open(path)
# # im.show()
# im = np.array(im)
#
# Image.fromarray(im).show()
# im = preprocess(im).astype(np.uint8)
# Image.fromarray(im).show()
# for _ in range(1):
#     Image.fromarray(random_channel_shift(im, 50., 2).astype(np.uint8)).show()
