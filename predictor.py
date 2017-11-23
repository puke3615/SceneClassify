import numpy as np


# predictions = model.predict(inputs)

class A:
    def __init__(self, value=1):
        self.value = value

    def predict(self, x, batch_size=32, verbose=0):
        return np.ones(x.shape) * self.value


DEFAULT_FUNC_PREDICT = 'predict'


class Predictor:
    def __init__(self, model, func_name=DEFAULT_FUNC_PREDICT):
        self.func_name = func_name
        self.funcs = model.__class__.__dict__
        self._check_method()
        self.func = self.funcs[self.func_name]
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.func(self.model, *args, **kwargs)

    def _check_method(self):
        if not self.funcs.__contains__(self.func_name):
            raise Exception('No function named "%s" found in model.' % self.func_name)


class IntegratedPredictor:
    POLICIES = ['avg', 'weight', 'label_weight']
    DEFAULT_POLICY = 'avg'

    def __init__(self, models=None, policy='avg'):
        self.models = models
        self.policy = policy if policy in self.POLICIES else self.DEFAULT_POLICY
        self.predictors = [Predictor(model) for model in models]

    def __call__(self, *args, **kwargs):
        if self.policy == 'avg':
            predictions = [predictor(*args, **kwargs) for predictor in self.predictors]
            prediction_summary = np.mean(predictions, axis=0)
        elif self.policy == 'weight':
            raise NotImplementedError('')
        elif self.policy == 'label_weight':
            raise NotImplementedError('')
        else:
            raise Exception('No support.')
        return prediction_summary


if __name__ == '__main__':
    single_predictor = Predictor(A(8))
    print(single_predictor(np.ones([1, 2])))

    integrated_predictor = IntegratedPredictor([
        A(),
        A(2),
        A(3),
        A(4),
    ])
    prediction = integrated_predictor(x=np.zeros([1, 3, 4]))
    print(prediction)
    print(prediction.shape)
