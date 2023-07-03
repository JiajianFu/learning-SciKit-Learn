from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #k临近分类

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

##print(iris_X[:2, :]) #打印出两个例子，具体是四项花的属性
##print(iris_y) #花的三个类别，分别是0，1，2

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3) #把所有数据随机分成用于学习的和测试的，不会出现人为的误差。
#测试数据占总数据的30%

##print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train) #所有训练步骤都在这一句
print(knn.predict(X_test))#用属性预测值
print(y_test)
