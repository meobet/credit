from __future__ import print_function, division
import argparse

from support import load, split_cv, save_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--x-file", type=str, help="X npy file")
    parser.add_argument("-y", "--y-file", type=str, help="Y npy file")
    parser.add_argument("-k", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("-cv", "--cv-file", type=str, help="Output cross-validation file")
    args = parser.parse_args()
    x, y, _ = load(args.x_file, args.y_file)
    save_cv(split_cv(x, y, args.k), args.cv_file)