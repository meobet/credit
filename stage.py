from __future__ import print_function, division

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from metrics import mean_absolute_error

from xgboost import XGBClassifier
from selection import select_xy, convert_xy
from support import load_cv, run_test, resample, apply_cv
from keras_classifier import LSTMClassifier, CNNClassifier
from sk_classifier import SkClassifier
from staged_classifier import Type0v3Classifier, Type0v123Classifier, AdjustedClassifier, StructuredClassifier2

import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main(x_train, x_test, y_train, y_test):
    result = []

    x_train_, y_train_ = x_train, y_train
    x_test_, y_test_ = x_test, y_test
    # x_train_, y_train_ = resample(RandomOverSampler(), x_train_, y_train_)
    # x_train_, y_train_ = select_xy(x_train_, y_train_, [1, 2])
    # x_train_, y_train_ = convert_xy(x_train_, y_train_, [0, 0, 1, 0])
    # x_test_, y_test_ = select_xy(x_test_, y_test_, [1, 2])
    # x_test_, y_test_ = convert_xy(x_test, y_test, [0, 0, 1, 0])

    # classifier = AdjustedClassifier(SkClassifier(XGBClassifier(), RandomOverSampler()), adjust=[-0.15, 0.])

    # x_train_, x_val, y_train_, y_val = train_test_split(x_train_, y_train_, test_size=0.1, stratify=y_train_)
    # classifier = AdjustedClassifier(Type0v123Classifier(base=(SkClassifier(XGBClassifier(), RandomOverSampler())),
    #                                                     aux=[(SkClassifier(XGBClassifier(), RandomOverSampler()))]),
    #                                 adjust=np.array([-0.5, 0, 0.15, 0]))
    classifier = StructuredClassifier2(base=AdjustedClassifier(SkClassifier(XGBClassifier()), adjust=[-0.25, 0]),
                                       aux=[SkClassifier(XGBClassifier()),
                                            SkClassifier(XGBClassifier())],
                                       sampler=RandomOverSampler())

    classifier.fit(x_train_, y_train_)

    # print(classifier.adjust)
    # result.append(classifier.adjust)

    print(classification_report(y_true=y_test_, y_pred=classifier.predict(x_test_)))
    result.append(recall_score(y_true=y_test_, y_pred=classifier.predict(x_test_), average=None))
    result.append(confusion_matrix(y_true=y_test_, y_pred=classifier.predict(x_test_)))

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

    # classifier = Type0v123Classifier(base=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
    #                                  aux=[SkClassifier(XGBClassifier(), sampler=RandomOverSampler())])
    # classifier.fit(x_train, y_train)
    #
    # y = classifier.predict(x_test)
    # print(confusion_matrix(y_true=y_test, y_pred=y))
    # print(classification_report(y_true=y_test, y_pred=y))
    # print(mean_absolute_error(y_true=y_test, y_pred=y))
    # result.append(confusion_matrix(y_true=y_test, y_pred=y)[1])
    # result.append(confusion_matrix(y_true=y_test, y_pred=y)[2])

    print("__________________")

    return result


if __name__ == "__main__":
    apply_cv("data/cv5.npz", main, True)
