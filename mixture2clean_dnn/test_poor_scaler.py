import numpy as np
from sklearn import preprocessing

from prepare_data import PoorScaler


a = np.array([[1., 2.], [9., 4.], [1., 1.]])

scaler = PoorScaler().fit(a)

assert (scaler.mean_== np.mean(a, 0)).any()
assert (scaler.scale_ == np.std(a, 0)).any()

scaler_a = scaler.transform(a)
correct =  preprocessing.StandardScaler().fit(a).transform(a)
assert (scaler_a == correct).any()
