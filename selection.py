import numpy as np
import sklearn.utils as utils


def select_y(y, values, shuffle=True):
    result = np.concatenate([np.where(y == value)[0] for value in values])
    if shuffle:
        result = utils.shuffle(result)
    return result


def select_xy(x, y, y_values, shuffle=True):
    index = select_y(y, y_values, shuffle)
    return x[index], y[index]


def convert_xy(x, y, y_list, shuffle=True):
    x_result = []
    y_result = []
    for i, value in enumerate(y_list):
        index = select_y(y, i)
        x_result.append(x[index])






