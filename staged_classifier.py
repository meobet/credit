import numpy as np
from selection import select_xy


class AugmentedClassifier(object):
    def __init__(self, base_classifier, aux_classifier):
        self.model = base_classifier
        self.aux = aux_classifier

    def fit(self, x, y, sample_weight=None):
        x_, y_ = select_xy(x, y, (0, 3))
        self.aux.fit(x_, y_)
        input_dim = x.shape[-1]
        x = np.hstack([np.repeat(self.aux.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim), x])
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        input_dim = x.shape[-1]
        x = np.hstack([np.repeat(self.aux.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim), x])
        return self.model.predict(x)

    def predict_proba(self, x):
        input_dim = x.shape[-1]
        x = np.hstack([np.repeat(self.aux.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim), x])
        return self.model.predict_proba(x)

    def save(self, filename):
        self.model.save(filename)
        self.aux.save(filename + ".aux")

    def load(self, filename):
        self.model.load(filename)
        self.aux.load(filename + ".aux")


class ThresholdClassifier(object):
    def __init__(self, base_classifier, threshold=None):
        self.model = base_classifier
        self.threshold = threshold

    def fit(self, x, y, sample_weight=None):
        x_, y_ = select_xy(x, y, (0, 3))
        self.model.fit(x_, y_, sample_weight=sample_weight)

        if self.threshold is None:
            xs = [select_xy(x, y, [i])[0] for i in range(4)]
            ps = [self.model.predict_proba(x)[:, 1] for x in xs]
            lo = [np.percentile(p, 20) for p in ps]
            hi = [np.percentile(p, 80) for p in ps]
            self.threshold = np.array([(l + h) / 2 for l, h in zip(lo[1:] + [1.], hi)])

    def predict(self, x):
        return np.array([np.argmax(y < self.threshold) for y in self.model.predict_proba(x)[:, 1]])

    def predict_proba(self, x):
        raise NotImplementedError()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)