import numpy as np


def mock(shape, value=1):
    return np.ones(shape) * value


def out(value, name=None):
    if not name:
        info = value.__class__.__dict__
        if info.__contains__('name'):
            name = info['name']
    if isinstance(value, list):
        num = len(value)
        shape = (num,) + value[0].shape
    else:
        shape = value.shape
    print('%s: %s' % (name, str(shape)))


def integrated_prob(predictions, mode, standard=False, **kwargs):
    predictions = np.array(predictions)
    result = None
    if mode == 'avg':
        result = np.mean(predictions, axis=0)
    elif mode == 'model_weight':
        c_ns = kwargs['model_weight']
        result = np.sum(c_n * p_nj for c_n, p_nj in zip(c_ns, predictions))
    elif mode == 'label_weight':
        c_njs = kwargs['label_weight']
        result = np.sum(c_nj * p_nj for c_nj, p_nj in zip(c_njs, predictions))
    elif mode == 'ada_boost':
        c_ns = kwargs['model_weight']
        c_njs = kwargs['label_weight']
        alphas = [np.log(c_n / (1 - c_n + 1e-6)) / 2 for c_n in c_ns]
        result = np.sum(c_nj * p_nj * alpha for alpha, c_nj, p_nj in zip(alphas, c_njs, predictions))
    if result is None:
        raise 'Not support for mode named "%s".' % mode
    elif standard:
        denominators = np.sum(result, axis=1)
        result = [r / denominator for r, denominator in zip(result, denominators)]
    return result


batch_size = 32
n_class = 80
n_model = 5

if __name__ == '__main__':
    ps = [mock([batch_size, n_class], i + 1) for i in range(n_model)]
    out(ps, 'ps')
    m_weights = mock([n_model])
    out(m_weights, 'm_weights')
    la_weights = mock([n_model, n_class])
    out(la_weights, 'la_weights')

    print('-' * 50)

    out(integrated_prob(ps, 'avg'), 'avg')
    out(integrated_prob(ps, 'model_weight', model_weight=m_weights), 'model_weight')
    out(integrated_prob(ps, 'label_weight', label_weight=la_weights), 'label_weight')
    out(integrated_prob(ps, 'ada_boost', model_weight=m_weights, label_weight=la_weights), 'ada_boost')
