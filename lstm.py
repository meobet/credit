import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam


class LSTMClassifier(object):
    def __init__(self,
              lstm_dim=200, dense_dims=[200, 100],
              batch_size=128, epochs=5, lr=1e-3,
              validation=0., sample_weight=None,
              dropout=0., activation_type="relu", cell_type=LSTM):
        self.lstm_dim = lstm_dim
        self.dense_dims = dense_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.validation = validation
        self.dropout = dropout
        self.activation_type = activation_type
        self.cell_type = cell_type
        self.verbose = 0

    def build(self, input_dim, timesteps, output_dim):
        x = Input(shape=(timesteps, input_dim))
        h = x
        h = self.cell_type(self.lstm_dim)(h)
        # h = Flatten()(h)
        for dim in self.dense_dims:
            h = Dense(dim, activation=self.activation_type)(h)
            if self.dropout > 0.:
                h = Dropout(self.dropout)(h)
        y = Dense(output_dim, activation="softmax")(h)
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=Adam(lr=self.lr), loss="sparse_categorical_crossentropy")

    def fit(self, x, y, sample_weight=None):
        self.build(input_dim=x.shape[2], timesteps=x.shape[1], output_dim=np.max(y) + 1)
        return self.model.fit(x, y,
                              batch_size=self.batch_size,
                              epochs=self.epochs,
                              validation_split=self.validation,
                              sample_weight=sample_weight,
                              verbose=self.verbose)

    def predict(self, x):
        probs = self.model.predict(x, batch_size=self.batch_size, verbose=self.verbose)
        return np.argmax(probs, axis=1)
