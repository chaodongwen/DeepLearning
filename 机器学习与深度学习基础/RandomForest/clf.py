import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 10)
col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv('iris.data', header=None, names=col_names)
print(data)
# x = data.iloc[:, :4]
# y = data.iloc[:, 4]
# x = data[['sepal length', 'sepal width']]
x = data[col_names[:4]]
y = data[col_names[4]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# model = SVC(C=1.0, kernel='rbf', gamma=0.01)
model = GridSearchCV(SVC(), cv=5, param_grid={
    'C': np.logspace(-4, 0, 20),
    'gamma': np.logspace(-2, -1, 10)
})
model.fit(x_train, y_train)
print('最优参数：', model.best_params_)
y_train_pred = model.predict(x_train)
print('训练集正确率：', accuracy_score(y_train, y_train_pred))
y_test_pred = model.predict(x_test)
print('测试集正确率：', accuracy_score(y_test, y_test_pred))
