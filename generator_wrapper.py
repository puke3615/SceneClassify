import Queue

from generator import *
from config import *
import random
import time


class BatchFetcherThread(threading.Thread):
    def __init__(self, func_next, queue, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(BatchFetcherThread, self).__init__(group, target, name, args, kwargs, verbose)
        self.func_next = func_next
        self.queue = queue

    def run(self):
        while True:
            batch = self.func_next()
            self.queue.put(batch)


class GeneratorWrapper(DirectoryIterator):
    def __init__(self, directory, image_data_generator, target_size=(256, 256), color_mode='rgb', classes=None,
                 class_mode='categorical', batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png', follow_links=False, crop_mode=None, batch_handler=None,
                 multi_thread=False, threads=5, queue_size=10, log=False):
        super(GeneratorWrapper, self).__init__(directory, image_data_generator, target_size, color_mode, classes,
                                               class_mode, batch_size, shuffle, seed, data_format, save_to_dir,
                                               save_prefix, save_format, follow_links, crop_mode, batch_handler)
        self.log = log
        self.threads = threads
        self.queue_size = queue_size
        self.multi_thread = multi_thread
        if self.multi_thread:
            self._queue = Queue.Queue(queue_size)
            self._auto_add(threads)

    def _auto_add(self, threads):
        for _ in range(threads):
            BatchFetcherThread(lambda: DirectoryIterator.next(self), self._queue).start()

    def next(self):
        if self.multi_thread:
            if self.log:
                print('The queue state is %d/%d.' % (self._queue.qsize(), self.queue_size))
            return self._queue.get()
        return super(GeneratorWrapper, self).next()


if __name__ == '__main__':
    start = time.time()

    multi_thread = True
    wrapper = GeneratorWrapper(
        PATH_TRAIN_IMAGES,
        ImageDataGenerator(),
        multi_thread=multi_thread,
        threads=10,
        log=True,
    )
    for i, batch in enumerate(wrapper):
        if i >= 100:
            break
        time.sleep(random.random() * 0.5)
    print 'Time takes %f second.' % (time.time() - start)
