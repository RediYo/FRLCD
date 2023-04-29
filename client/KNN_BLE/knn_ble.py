# Import necessary libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from client.ble_dataset import load_ble

# 加载数据集
ble = load_ble()
X = ble.data
y = ble.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
X_train = X_train.astype('float')
y_train = y_train.astype('int')
X_test = X_test.astype('float')
y_test = y_test.astype('int')
# Fit the classifier to the data
knn.fit(X_train, y_train)
# Predict the classes of the testing data
y_pred = knn.predict(X_test)
# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
