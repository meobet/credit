from __future__ import print_function, division

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lstm import LSTMClassifier
from keras.layers import LSTM, GRU

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from support import resample, get_vae_features, run_test, run_cv



# transform data with imblearn resampler
def prep_resample(x_train, x_test, y_train, y_test, sampler):
    if sampler is not None:
        x_train, y_train = resample(sampler, x_train, y_train)
    return x_train, x_test, y_train, y_test


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
    classifiers = [LSTMClassifier(epochs=20, dropout=0.1, cell_type=GRU),
                   XGBClassifier]
    flatten = [False, True] # 3D classifier doesn't need flat input
    for resampler in [None, SMOTE(), ADASYN(), RandomOverSampler()]:
        print(resampler)
        prep = lambda *args: prep_resample(*args, sampler=resampler)
        run_cv("data/cv5.npz",
               preprocessor=prep,
               classifiers=classifiers,
               flatten=flatten)
