import numpy as np
from selection import select_xy, convert_xy
from support import resample

class AugmentedClassifier(object):
    def __init__(self, base, aux, sampler=None):
        self.model = base
        self.aux = aux
        self.name = type(self).__name__ + "[" + base.name + "]"
        self.sampler = sampler

    def fit(self, x, y, sample_weight=None):
        if self.sampler is not None:
            x, y = resample(self.sampler, x, y)
        self.fit_aux(x, y)
        input_dim = x.shape[-1]
        x_aux = [np.repeat(classifier.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim)
                 for classifier in self.aux]
        x = np.hstack(x_aux + [x])
        self.model.fit(x, y, sample_weight=sample_weight)

    def predict(self, x):
        input_dim = x.shape[-1]
        x_aux = [np.repeat(classifier.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim)
                 for classifier in self.aux]
        x = np.hstack(x_aux + [x])
        return self.model.predict(x)

    def predict_proba(self, x):
        input_dim = x.shape[-1]
        x_aux = [np.repeat(classifier.predict_proba(x)[:, 1], input_dim).reshape(-1, 1, input_dim)
                 for classifier in self.aux]
        x = np.hstack(x_aux + [x])
        return self.model.predict_proba(x)

    def visualize(self, x, y):
        pass

    def save(self, filename):
        self.model.save(filename)
        for i, classifier in enumerate(self.aux):
            classifier.save(filename + "." + str(i) + ".aux")

    def load(self, filename):
        self.model.load(filename)
        for i, classifier in enumerate(self.aux):
            classifier.load(filename + "." + str(i) + ".aux")


class ThresholdClassifier(object):
    def __init__(self, base, threshold=None):
        self.model = base
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


class Type0v3Classifier(AugmentedClassifier):
    def fit_aux(self, x, y):
        assert len(self.aux) == 1
        self.aux[0].fit(*select_xy(x, y, [0, 3]))


class Type0v123Classifier(AugmentedClassifier):
    def fit_aux(self, x, y):
        assert len(self.aux) == 1
        self.aux[0].fit(*convert_xy(x, y, [0, 1, 1, 1]))


class Type01v23Classifier(AugmentedClassifier):
    def fit_aux(self, x, y):
        assert len(self.aux) == 1
        self.aux[0].fit(*convert_xy(x, y, [0, 0, 1, 1]))


class Combo1Classifier(AugmentedClassifier):
    def fit_aux(self, x, y):
        assert len(self.aux) == 3
        self.aux[0].fit(*select_xy(x, y, [0, 3]))
        self.aux[1].fit(*convert_xy(x, y, [0, 1, 1, 1]))
        self.aux[2].fit(*convert_xy(x, y, [0, 0, 1, 1]))


class StructuredClassifier(AugmentedClassifier):
    def fit(self, x, y, sample_weight=None):
        if self.sampler is not None:
            x, y = resample(self.sampler, x, y)
        self.aux[0].fit(*convert_xy(x, y, [0, 1, 1, 1]))
        self.model.fit(*select_xy(x, y, [1, 2, 3]))

    def predict(self, x):
        y_0 = self.aux[0].predict(x)
        y_123 = self.model.predict(x)
        return np.where(y_0 == 0, y_0, y_123)

    def predict_proba(self, x):
        raise NotImplementedError()
        
        
class StructuredClassifier2(AugmentedClassifier):
    def fit(self, x, y, sample_weight=None):
        if self.sampler is not None:
            x, y = resample(self.sampler, x, y)
        self.aux[0].fit(*convert_xy(x, y, [0, 1, 1, 1]))
        self.aux[1].fit(*convert_xy(*select_xy(x, y, [1, 2, 3]), [0, 2, 2, 3]))
        self.model.fit(*select_xy(x, y, [1, 2]))

    def predict(self, x):
        y_0 = self.aux[0].predict(x)
        y_3 = self.aux[1].predict(x)
        y_12 = self.model.predict(x)
        return np.where(y_0 == 0, y_0, np.where(y_3 == 3, y_3, y_12))

    def predict_proba(self, x):
        raise NotImplementedError()


class AdjustedClassifier(object):
    def __init__(self, base, adjust=np.zeros(4, dtype=float), validation=None):
        self.model = base
        self.adjust = adjust
        self.validation = validation
        self.name = type(self).__name__ + "[" + base.name + "]"

    def fit(self, x, y, sample_weight=None):
        self.model.fit(x, y, sample_weight=sample_weight)
        self.min_label = np.min(y)
        if self.validation is not None:
            self.adjust = np.zeros(4, dtype=float)
            x_v, y_v = self.validation
            probs = self.model.predict_proba(x_v)
            activation_0 = probs[:, 0] - np.max(probs[:, 1:], axis=1)
            true_0 = activation_0[np.where(y_v == 0)]
            self.adjust[0] = -np.percentile(true_0, 20)

    def predict(self, x):
        probs = self.model.predict_proba(x) + self.adjust
        return np.argmax(probs, axis=1) + self.min_label

    def predict_proba(self, x):
        raise NotImplementedError()

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)