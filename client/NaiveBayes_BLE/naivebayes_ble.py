from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from client.ble_dataset import load_ble

# Load data
ble = load_ble()
X = ble.data
y = ble.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 解决 ValueError: Unknown label type: 'unknown'
X_train = X_train.astype('float')
y_train = y_train.astype('int')
X_test = X_test.astype('float')
y_test = y_test.astype('int')

# Create Naive Bayes classifier
clf = GaussianNB()

# Train classifier on data
clf.fit(X_train, y_train)

# Make predictions on new data
y_pred = clf.predict(X_test)

# Evaluate accuracy of predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
