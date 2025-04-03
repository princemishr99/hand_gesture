import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

data_dict = pickle.load(open('./data.pickle','rb'))

max_length = max(len(seq) for seq in data_dict['data'])

# Standardize the length by padding with a placeholder (e.g., None or a specific value)
standardized_data = [seq + [None] * (max_length - len(seq)) for seq in data_dict['data']]

# Convert to NumPy array
dat = np.asarray(standardized_data)

data = np.asarray(dat)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score*100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
