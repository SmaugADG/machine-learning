from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def knn_iris():
    '''
    对鸢尾花进行分类
    :return:
    '''
    # 1）获取数据
    iris=load_iris()
    # 2） 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=23)
    # 3） 特征工程--标准化处理
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4）KNN算法评估
    estimator=KNeighborsClassifier()
    estimator.fit(x_train,y_train)
    # 5）模型评估
    # 方法一:直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("比对结果：\n",y_test==y_predict)
    # 方法二：季孙准确率
    score=estimator.score(x_test,y_test)
    print("准确率：\n",score)

    return None

def knn_iris_gscv():
    '''
    对鸢尾花进行分类
    模型调优
    :return:
    '''
    # 1）获取数据
    iris=load_iris()
    # 2） 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=23)
    # 3） 特征工程--标准化处理
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 4）KNN算法评估
    estimator=KNeighborsClassifier()

    # 添加网格搜索和交叉验证
    # 准备参数
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)

    estimator.fit(x_train,y_train)
    # 5）模型评估
    # 方法一:直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("比对结果：\n",y_test==y_predict)
    # 方法二：季孙准确率
    score=estimator.score(x_test,y_test)
    print("准确率：\n",score)

    # 训练验证集的结果
    print("最佳参数：\n",estimator.best_params_)
    print("在交叉验证当中验证的最好结果：", estimator.best_score_)
    print("模型K值是：", estimator.best_estimator_)
    print("交叉验证的结果为：", estimator.cv_results_)

    return None


if __name__ == '__main__':
    # 对鸢尾花进行分类
    # knn_iris()

    # 0.9473684210526315
    # 0.9736842105263158
    # 模型调优和交叉验证
    knn_iris_gscv()