from __future__ import print_function, division

import numpy as np
from collections import defaultdict

def mean_absolute_error(y_true, y_pred):
    class_ap = defaultdict(float)
    class_counts = defaultdict(int)
    for target, pred in zip(y_true, y_pred):
        class_ap[target] += abs(target - pred)
        class_counts[target] += 1
    return np.array([class_ap[target] / class_counts[target] for target in sorted(class_ap.keys())])



