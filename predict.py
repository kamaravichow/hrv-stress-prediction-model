import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

testFile = pd.read_csv('./dataset/test.csv').drop(columns="datasetId")

test_samples = testFile.drop(columns='condition').to_numpy()
test_labels = testFile['condition'].to_numpy()

scaler = MinMaxScaler(feature_range=(0, 1))
test_samples = scaler.fit_transform(test_samples)

one_hot_encoder = OneHotEncoder(categories='auto')
test_labels = one_hot_encoder.fit_transform(test_labels.reshape(-1, 1)).toarray()

print("Test lables shape : " + test_labels.shape)

model = keras.models.load_model('./model.h5')

print(model.summary())

predictions = model.predict(test_samples)

lables = one_hot_encoder.inverse_transform(test_labels)
print(predictions)

result = np.append(predictions, lables)

np.savetxt('predictions.csv', predictions, delimiter=",")
file = open("lables.txt", "w+")
file.write(str(lables))
file.close()
print(result)
