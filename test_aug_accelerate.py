# coding=utf-8
import multiprocessing
from PIL import Image
import numpy as np
import threading
import time
import os

import im_utils
from config import *

"""
测试单线程、多线程、多进程的时间差异
"""

COUNT = 32


def get_files(size, dir=os.path.join(PATH_TRAIN_IMAGES, '00')):
    return [os.path.join(dir, sub) for sub in np.random.choice(os.listdir(dir), size, replace=False)]


files = get_files(COUNT)
images = np.array(
    [np.asarray(Image.open(f).resize((299, 299))) for f in files],
    dtype=np.uint8
)


def p(*args):
    print(args)


def summary(func, n_test=1, name=None, **kwargs):
    if not name:
        name = func.__name__
    times = 0
    for i in range(n_test):
        start = time.time()
        func(**kwargs)
        times += (time.time() - start)
        if n_test > 1:
            print('Test: %d/%d' % (i + 1, n_test))
    print('%s takes %.2f seconds.' % (name, times / n_test))


def invoke(func, invoke_count=1, task_mode=None, log=False):
    times = []

    def call(start, end):
        times.append([start, end])

    for i in range(invoke_count):
        if task_mode == 'Thread':
            threading.Thread(target=lambda: record(call, func)).start()
        elif task_mode == 'Process':
            pool.apply_async(record, args=(None, func), callback=lambda x: call(*x))
        else:
            record(call, func)
        if log:
            print('Invoke: %d/%d' % (i + 1, invoke_count))
    while len(times) != invoke_count:
        time.sleep(1e-6)
    return times


def record(call, func):
    start = time.time()
    func()
    end = time.time()
    if callable(call):
        call(start, end)
    return start, end


def aug():
    var = [im_utils.aug_images([image])[0] for image in images]


class Pool:
    def __init__(self):
        self.pool = multiprocessing.Pool()

    def __enter__(self):
        return self.pool

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()


if __name__ == '__main__':
    TASK_MODE = [None, 'Thread', 'Process']
    with Pool() as pool:
        summary(
            lambda: summary(lambda: invoke(aug, 10, 'Process', True), n_test=10, name='Aug[%d]' % COUNT),
            name='Total'
        )
