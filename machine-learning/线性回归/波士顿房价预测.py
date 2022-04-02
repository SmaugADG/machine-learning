from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error

def liner1():
    '''
    正规方程
    :return:
    '''
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 预估器
    estimator=LinearRegression()
    estimator.fit(x_train,y_train)

    # 得出模型
    print("line1权重系数\n",estimator.coef_)
    print("line1偏置值：\n",estimator.intercept_)

    # 模型评估
    y_predict=estimator.predict(x_test)
    print(" line1预测房价:\n",y_predict)
    error=mean_squared_error(y_test,y_predict)
    print("line1均方误差：\n",error)

    return None


def liner2():
    '''
    梯度下降算法
    :return:
    '''
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 预估器
    estimator=SGDRegressor()
    estimator.fit(x_train,y_train)

    # 得出模型
    print("line2权重系数\n",estimator.coef_)
    print("line2偏置值：\n",estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("line2 预测房价:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("line2均方误差：\n", error)

    return None


def liner3():
    '''
    岭回归
    :return:
    '''
    # 获取数据
    boston = load_boston()

    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(boston.data,boston.target,random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)

    # 预估器
    estimator=Ridge()
    estimator.fit(x_train,y_train)

    # 得出模型
    print("line3权重系数\n",estimator.coef_)
    print("line3偏置值：\n",estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("line3 预测房价:\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("line3均方误差：\n", error)

    return None


if __name__ == '__main__':
    # 正规方程
    liner1()

    # 梯度下降
    liner2()

    # 岭回归
    liner3()
