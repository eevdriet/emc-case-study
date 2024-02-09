import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

from emc.classifiers import Classifier
from emc.model import Label

class LSTMClassifier(Classifier):
    def _preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        return super()._preprocess(features)
    
    def _train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_samples = len(X_train)
        simulation_dim = X_train[0].shape
        n_time_steps = simulation_dim[0]
        n_features = simulation_dim[1]

        self.model = Sequential()
        self.model.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(n_time_steps, n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(20, activation='relu', return_sequences=True))
        self.model.add(LSTM(10, activation='relu'))
        self.model.add(Dense(1, activation='softmax'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)



    def _test(self, X_test: np.ndarray, y_test: np.ndarray):
        return self.model.predict(X_test), y_test