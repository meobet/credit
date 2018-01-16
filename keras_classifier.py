import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from support import resample

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
import keras.regularizers as reg


class KerasClassifier(object):
    def __init__(self,
                 batch_size=128, epochs=5,
                 validation=0., criterion="val_acc", patience=5,
                 verbose=0, sampler=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation = validation
        self.criterion = criterion
        self.patience = patience
        self.verbose = verbose
        self.checkpoint = "data/keras.{epoch:02d}.model"
        self.sampler = sampler

    def build(self, input_dim, timesteps, output_dim):
        raise NotImplementedError()

    def fit(self, x, y, sample_weight=None):
        self.build(input_dim=x.shape[2], timesteps=x.shape[1], output_dim=np.max(y) + 1)
        if self.validation > 0:
            if sample_weight is None:
                x, x_v, y, y_v = train_test_split(x, y, test_size=self.validation, stratify=y)
            else:
                x, x_v, y, y_v, sample_weight, _ = train_test_split(x, y, sample_weight,
                                                                    test_size=self.validation,
                                                                    stratify=y)
            if self.sampler is not None:
                x, y = resample(self.sampler, x, y)
                x_v, y_v = resample(self.sampler, x_v, y_v)
                sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            checkpoint = ModelCheckpoint(self.checkpoint, monitor=self.criterion, save_best_only=True)
            stopper = EarlyStopping(monitor="val_acc", patience=self.patience)
        return self.model.fit(x, y,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              validation_data=(x_v, y_v) if self.validation > 0 else None,
                              callbacks=[checkpoint, stopper] if self.validation > 0 else None,
                              sample_weight=sample_weight,
                              verbose=self.verbose)

    def predict(self, x):
        probs = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return np.argmax(probs, axis=1)

    def predict_proba(self, x):
        return self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)


class LSTMClassifier(KerasClassifier):
    def __init__(self,
                 lstm_dims, dense_dims,
                 dropout=0., activation_type="relu", cell_type=LSTM,
                 optimizer=Adam, lr=1e-3,
                 batch_size=128, epochs=5,
                 validation=0., patience=5, criterion="val_acc",
                 verbose=0, sampler=None):
        super(LSTMClassifier, self).__init__(batch_size=batch_size,
                                             epochs=epochs,
                                             validation=validation,
                                             patience=patience,
                                             criterion=criterion,
                                             verbose=verbose,
                                             sampler=sampler)
        self.lstm_dims = lstm_dims
        self.dense_dims = dense_dims
        self.dropout = dropout
        self.activation_type = activation_type
        self.cell_type = cell_type

        self.optimizer_type = optimizer
        self.lr = lr

        self.checkpoint = "data/lstm.{epoch:02d}.model"

    def build(self, input_dim, timesteps, output_dim):
        x = Input(shape=(timesteps, input_dim))
        h = self.cell_type(self.lstm_dims[0], return_sequences=True,
                           dropout=self.dropout, recurrent_dropout=self.dropout,
                           kernel_regularizer=reg.l2(1e-5), activity_regularizer=reg.l1(1e-5))(x)
        for dim in self.lstm_dims[1:-1]:
            h = self.cell_type(dim, return_sequences=True, dropout=self.dropout)(h)
        h = self.cell_type(self.lstm_dims[-1], return_sequences=False, dropout=self.dropout)(h)
        for dim in self.dense_dims:
            h = Dense(dim, activation=self.activation_type)(h)
            if self.dropout > 0.:
                h = Dropout(self.dropout)(h)
        y = Dense(output_dim, activation="softmax")(h)
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer_type(lr=self.lr),
                           loss="sparse_categorical_crossentropy",
                           metrics=["acc"])


class CNNClassifier(KerasClassifier):
    def __init__(self,
                 conv_dims, conv_kernels, pool_sizes, dense_dims,
                 dropout=0., activation_type="relu",
                 optimizer=Adam, lr=1e-3,
                 batch_size=128, epochs=5,
                 validation=0., patience=5, criterion="val_acc",
                 verbose=0, sampler=None):
        super(CNNClassifier, self).__init__(batch_size=batch_size,
                                            epochs=epochs,
                                            validation=validation,
                                            patience=patience,
                                            criterion=criterion,
                                            verbose=verbose,
                                            sampler=sampler)
        self.conv_dims = conv_dims
        self.conv_kernels = conv_kernels
        self.pool_sizes = pool_sizes
        self.dense_dims = dense_dims
        self.dropout = dropout
        self.activation_type = activation_type

        self.optimizer_type = optimizer
        self.lr = lr

        self.checkpoint = "data/lstm.{epoch:02d}.model"

    def build(self, input_dim, timesteps, output_dim):
        assert len(self.conv_dims) == len(self.conv_kernels) == len(self.pool_sizes)
        x = Input(shape=(timesteps, input_dim))
        h = x
        for dim, kernel, pool in zip(self.conv_dims, self.conv_kernels, self.pool_sizes):
            h = Conv1D(dim, kernel_size=kernel)(h)
            h = MaxPooling1D(pool)(h)
            if self.dropout > 0.:
                h = Dropout(self.dropout)(h)
        h = Flatten()(h)
        for dim in self.dense_dims:
            h = Dense(dim, activation=self.activation_type)(h)
            if self.dropout > 0.:
                h = Dropout(self.dropout)(h)
        y = Dense(output_dim, activation="softmax")(h)
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer_type(lr=self.lr),
                           loss="sparse_categorical_crossentropy",
                           metrics=["acc"])