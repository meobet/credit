from __future__ import print_function, division

import argparse
import numpy as np
from xgboost import XGBClassifier
from sk_classifier import SkClassifier
from keras_classifier import CNNClassifier
from staged_classifier import AugmentedClassifier, ThresholdClassifier

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from support import resample, get_vae_features, run_test, run_cv


# transform data with LSTM-VAE features and imblearn resampler
def prep_vae(x_train, x_test, y_train, y_test, sampler=None):
    x_train, x_test = get_vae_features(x_train, x_test,
                                       epochs=40,
                                       batch_size=128,
                                       intermediate_dim=1000,
                                       latent_dim=100)
    if sampler is not None:
        x_train, y_train = sampler.fit_sample(x_train, y_train)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Running cross-validation")
    parser.add_argument("-cv", "--cv-file", type=str, default="data/cv5.npz", help="Cross-validation data file")
    args = parser.parse_args()

    classifiers = [ThresholdClassifier(base_classifier=SkClassifier(XGBClassifier(), sampler=RandomOverSampler())),
                   AugmentedClassifier(base_classifier=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()),
                                       aux_classifier=SkClassifier(XGBClassifier(), sampler=RandomOverSampler()))]
    run_cv(args.cv_file, classifiers=classifiers)

