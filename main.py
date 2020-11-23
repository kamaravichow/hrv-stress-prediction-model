import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical

trainFile = pd.read_csv('./dataset/train.csv').drop(columns="datasetId")
testFile = pd.read_csv('./dataset/test.csv').drop(columns="datasetId")

# train
train_samples = trainFile.drop(columns='condition').to_numpy()
train_labels = trainFile['condition'].to_numpy()

# test
test_samples = testFile.drop(columns='condition').to_numpy()
test_labels = testFile['condition'].to_numpy()

# normalizing features
scaler = MinMaxScaler(feature_range=(0, 1))
train_samples = scaler.fit_transform(train_samples)
test_samples = scaler.fit_transform(test_samples)

# one-hot-encoding labels
one_hot_encoder = OneHotEncoder(categories='auto')
train_labels = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1)).toarray()
test_labels = one_hot_encoder.fit_transform(test_labels.reshape(-1, 1)).toarray()

# build the model
model = Sequential([
    Dense(34, input_shape=[34, ], activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

print(model.summary())

model.compile(Adam(lr=.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=10, shuffle=True, verbose=2)

model.save('model.h5')

predictions = model.predict(test_samples)

print(predictions)

np.savetxt('predictions.csv', test_samples, delimiter=",")
