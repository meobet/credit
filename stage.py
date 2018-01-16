from __future__ import print_function, division

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from selection import select_xy
from support import load_cv, run_test, resample
from keras_classifier import LSTMClassifier, CNNClassifier
from sk_classifier import SkClassifier
from staged_classifier import AugmentedClassifier

import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_cv("data/cv5.2.npz")[4]
    xs = [select_xy(x_train, y_train, [i])[0] for i in range(4)]

    x_train_03, y_train_03 = select_xy(x_train, y_train, [0, 3])
    x_test_03, y_test_03 = select_xy(x_test, y_test, [0, 3])

    # classifier = LSTMClassifier(lstm_dims=[100, 200],
    #                             dense_dims=[200, 100],
    #                             epochs=100,
    #                             validation=0.2,
    #                             dropout=0.1,
    #                             verbose=2,
    #                             patience=10,
    #                             sampler=RandomOverSampler())
    # classifier.fit(x_train, y_train)
    # classifier.save("data/lstm03.model")

    # classifier = CNNClassifier(conv_dims=[200, 200, 100, 100],
    #                            conv_kernels=[3, 3, 3, 3],
    #                            pool_sizes=[2, 2, 2, 2],
    #                            dense_dims=[50, 50],
    #                            epochs=100,
    #                            validation=0.2,
    #                            dropout=0.1,
    #                            verbose=2,
    #                            patience=10,
    #                            criterion="val_acc",
    #                            sampler=SMOTE())
    # classifier.fit(x_train, y_train)
    # classifier.save("data/cnn03.model")

    classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    classifier.fit(x_train, y_train)
    print(confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test)))
    print(classification_report(y_true=y_test, y_pred=classifier.predict(x_test)))

    print("__________________")

    # classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    # classifier.fit(x_train_03, y_train_03)

    # print(classification_report(y_true=y_train_03, y_pred=classifier.predict(x_train_03)))
    # print(classification_report(y_true=y_test_03, y_pred=classifier.predict(x_test_03)))

    # ps = [classifier.predict_proba(x)[:, 1] for x in xs]
    # for p in ps:
    #     print(np.percentile(p, 20), np.percentile(p, 80))
    # plt.boxplot(ps)
    # plt.show()

    # input_dim = x_train.shape[-1]
    # xp = classifier.predict_proba(x_train)[:, 1]
    # x_train = np.hstack([np.repeat(xp, input_dim).reshape(-1, 1, input_dim), x_train])
    # xp = classifier.predict_proba(x_test)[:, 1]
    # x_test = np.hstack([np.repeat(xp, input_dim).reshape(-1, 1, input_dim), x_test])
    #
    # classifier = SkClassifier(XGBClassifier(), sampler=RandomOverSampler())
    # classifier.fit(x_train, y_train)

    classifier = AugmentedClassifier(base_classifier=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
                                     aux_classifier=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()))
    classifier.fit(x_train, y_train)

    print(confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test)))
    print(classification_report(y_true=y_test, y_pred=classifier.predict(x_test)))
