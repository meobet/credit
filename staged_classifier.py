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
        self.model.fit(x, y, sample_weight)

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