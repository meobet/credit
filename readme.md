### Creating cross-validation data
To make a cross-validation data file with 5 splits, run:

    python make_cv.py -x x_npy_file -y y_npy_file -k 5 -cv output_file

This file can be loaded with support.load_cv().

### Running experiments
See the main loop in simple.py.

For the latest model, import class StructuredClassifier from staged_classifier.py and
use it like a normal sklearn classifier (but there is no predict_proba function).

More notes:

Both KerasClassifier and SkClassifier have the same interface, and can be constructed with an
imblearn resampler.

The AugmentedClassifier can wrap any KerasClassifier or SkClassifier.