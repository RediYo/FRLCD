from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from client.ble_dataset import load_ble

# 加载数据集
ble = load_ble()
X = ble.data
y = ble.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 创建一个SVM分类器
svm = SVC(kernel='linear', C=1.0, random_state=0)

# 在训练集上训练分类器
svm.fit(X_train, y_train)

# 在测试集上测试分类器
y_pred = svm.predict(X_test)

# 计算分类器的准确率
accuracy = accuracy_score(y_test, y_pred)

print("SVM的准确率为：", accuracy)
