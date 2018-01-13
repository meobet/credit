from __future__ import print_function, division

from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, recall_score

from vae import create_lstm_vae


# load original npy
def load(x_file, y_file):
    x = np.load(x_file)
    y = np.load(y_file).astype(int)
    label_counts = np.bincount(y)
    return x, y, label_counts


# cross-validation split
def split_cv(x, y, k=5):
    splitter = StratifiedKFold(n_splits=k)
    result = []
    for train, test in splitter.split(x, y):
        result.append((x[train], x[test], y[train], y[test]))
    return result


# save cross-validation to npz
def save_cv(result, filename):
    arrays = [x for i in result for x in i]
    names = [name + "_" + str(i) for i in range(len(result)) for name in ["x_train", "x_test", "y_train", "y_test"]]
    with open(filename, "wb") as target:
        np.savez(target, **dict(zip(names, arrays)))


# load cross-validation npz
def load_cv(filename):
    data = np.load(filename)
    result = []
    for i in range(len(data.files) // 4):
        result.append(tuple(data[name + "_" + str(i)] for name in ["x_train", "x_test", "y_train", "y_test"]))
    return result


################################################################################################################


# apply imblearn resamplers to 3D x and 1D y
def resample(sampler, x, y):
    x_, y_ = sampler.fit_sample(x.reshape(x.shape[0], -1), y)
    return x_.reshape(-1, *x.shape[1:]), y_


# train LSTM-VAE on x_train and calculate features for x_train and x_test
def get_vae_features(x_train, x_test=None,
                     batch_size=128,
                     epochs=20,
                     intermediate_dim=500,

                     latent_dim=50):
    def pad(x):
        batch_pad = batch_size - x.shape[0] % batch_size
        return np.vstack((x, x[:batch_pad]))

    vae, encoder, generator = create_lstm_vae(input_dim=x_train.shape[2],
                                              timesteps=x_train.shape[1],
                                              batch_size=batch_size,
                                              intermediate_dim=intermediate_dim,
                                              latent_dim=latent_dim)
    x_train_pad = pad(x_train)
    vae.fit(x_train_pad, x_train_pad, batch_size=batch_size, epochs=epochs)
    x_train = encoder.predict(x_train_pad, batch_size=batch_size)[:x_train.shape[0]]

    if x_test is None:
        return x_train
    else:
        x_test_pad = pad(x_test)
        x_test = encoder.predict(x_test_pad, batch_size=batch_size)[:x_test.shape[0]]
        return x_train, x_test


# return: confusion matrix and recall scores
def run_test(x_train, x_test, y_train, y_test, classifier, flatten):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    classifier.fit(x_train, y_train, sample_weight=sample_weight)
    y_pred = classifier.predict(x_test)
    return (confusion_matrix(y_true=y_test, y_pred=y_pred),
                   recall_score(y_true=y_test, y_pred=y_pred, labels=np.arange(4), average=None))


# run test_function on cross-validation data
def run_cv(filename, preprocessor, classifiers, flatten=True):
    data = load_cv(filename)
    result = defaultdict(list)
    for iteration in data:
        datasets = preprocessor(*iteration)
        if flatten is True or flatten is False:
            flatten = [flatten] * len(classifiers)
        for classifier, flat in zip(classifiers, flatten):
            result[classifier].append(run_test(*datasets, classifier=classifier, flatten=flat))
    for classifier in classifiers:
        print(type(classifier))
        output = result[classifier]
        for i in range(len(output[0])):
            if output[0][i].ndim == 2:
                print(np.sum([score[i] for score in output], axis=0))
            else:
                print(np.mean([score[i] for score in output], axis=0))
        print()
