### Creating cross-validation data
To make a cross-validation data file with 5 splits, run:

    python make_cv.py -x x_npy_file -y y_npy_file -k 5 -cv output_file

This file can be loaded with support.load_cv().

### Running experiments
See the main loop in simple.py.

Choose a list classifiers (sklearn format).

Make a list of flatten variables, one value for each classifier.
Flatten value is True if the classifier takes 2D input, False if the classifier takes 3D input (like LSTM).

Choose resamplers (imblearn format) and write a preprocessor function - see prep_resampler() in simple.py.

Run support.run_cv(), which returns the confusion matrix and recall for each class.