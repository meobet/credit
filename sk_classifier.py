import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.externals import joblib
from support import resample


class SkClassifier(object):
    def __init__(self, base_classifier, sampler=None):
        self.model = base_classifier
        self.sampler = sampler
        self.name = type(base_classifier).__name__

    def fit(self, x, y, sample_weight=None):
        if self.sampler is not None:
            x, y = resample(self.sampler, x, y)
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return self.model.fit(x.reshape(x.shape[0], -1), y, sample_weight=sample_weight)

    def predict(self, x):
        return self.model.predict(x.reshape(x.shape[0], -1))

    def predict_proba(self, x):
        return self.model.predict_proba(x.reshape(x.shape[0], -1))

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)
