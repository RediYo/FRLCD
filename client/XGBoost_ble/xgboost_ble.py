from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from client.XGBoost_ble.ble_dataset import load_ble

# 加载数据集
ble = load_ble()
X = ble.data
y = ble.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 构建XGBoost模型
xgb = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=10)
xgb.fit(X_train, y_train)

# 预测并计算RMSE
y_pred = xgb.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
