from __future__ import print_function, division

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, classification_report
from metrics import mean_absolute_error

from xgboost import XGBClassifier
from selection import select_xy, convert_xy
from support import load_cv, run_test, resample
from keras_classifier import LSTMClassifier, CNNClassifier
from sk_classifier import SkClassifier
from staged_classifier import Type0v3Classifier, Type0v123Classifier, Combo1Classifier

import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_cv("data/cv5.2.npz")[4]
    xs = [select_xy(x_train, y_train, [i])[0] for i in range(4)]

    # x_train_, y_train_ = select_xy(x_train, y_train, [1, 2])
    # x_test_, y_test_ = select_xy(x_test, y_test, [1, 2])

    x_train_, y_train_ = convert_xy(x_train, y_train, [0, 1, 1, 0])
    x_test_, y_test_ = convert_xy(x_test, y_test, [0, 1, 1, 0])

    # classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    # classifier.fit(x_train, y_train)
    # print(confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test)))
    # print(classification_report(y_true=y_test, y_pred=classifier.predict(x_test)))

    print("__________________")

    classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    classifier.fit(x_train_, y_train_)

    print(classification_report(y_true=y_train_, y_pred=classifier.predict(x_train_)))
    print(classification_report(y_true=y_test_, y_pred=classifier.predict(x_test_)))
    #
    # ps = [classifier.predict_proba(x)[:, 1] for x in xs]
    # for p in ps:
    #     print(np.percentile(p, 20), np.percentile(p, 80))
    # plt.boxplot(ps)
    # plt.show()
    #
    # input_dim = x_train.shape[-1]
    # xp = classifier.predict_proba(x_train)[:, 1]
    # x_train = np.hstack([np.repeat(xp, input_dim).reshape(-1, 1, input_dim), x_train])
    # xp = classifier.predict_proba(x_test)[:, 1]
    # x_test = np.hstack([np.repeat(xp, input_dim).reshape(-1, 1, input_dim), x_test])
    #
    # classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    # classifier.fit(x_train, y_train)

    # classifier = Combo1Classifier(base=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
    #                               aux=[SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
    #                                    SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
    #                                    SkClassifier(XGBClassifier(), sampler=RandomOverSampler())])
    # classifier.fit(x_train, y_train)
    #
    # y = classifier.predict(x_test)
    # print(confusion_matrix(y_true=y_test, y_pred=y))
    # print(classification_report(y_true=y_test, y_pred=y))
    # print(mean_absolute_error(y_true=y_test, y_pred=y))
