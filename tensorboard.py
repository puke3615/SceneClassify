from keras.callbacks import TensorBoard


class StepTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 init_steps=None,
                 skip_steps=100):
        super(StepTensorBoard, self).__init__(log_dir, histogram_freq, write_graph, write_images, embeddings_freq,
                                              embeddings_layer_names, embeddings_metadata)
        self.steps = init_steps or 0
        self.skip_steps = skip_steps

    def on_batch_end(self, batch, logs=None):
        self.steps += 1
        if self.steps == 1 or self.steps % self.skip_steps == 0:
            super(StepTensorBoard, self).on_epoch_end(self.steps, logs)

    def on_epoch_end(self, epoch, logs=None):
        pass
