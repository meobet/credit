### Creating cross-validation data
To make a cross-validation data file with 5 splits, run:

    python make_cv.py -x x_npy_file -y y_npy_file -k 5 -cv output_file

This file can be loaded with support.load_cv().

### Running experiments
See the main loop in simple.py.

The classifiers should be any type in keras_classifier.py (they are subclasses of KerasClassifier)
or SkClassifier (which should be constructed with a sklearn or xgboost classifier).

Both KerasClassifier and SkClassifier have the same interface, and can be constructed with an
imblearn resampler.

The AugmentedClassifier can wrap any KerasClassifier or SkClassifier.