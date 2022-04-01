from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    '''
    使用朴素贝叶斯算法对新闻进行分类
    :return:
    '''
    # 获取数据
    news=fetch_20newsgroups(subset="all")
    # 划分数据集
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target)
    # 特征工程：文本特征抽取-tfidf
    transfer=TfidfVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    # 朴素贝叶斯算法预估
    estimator=MultinomialNB()
    estimator.fit(x_train,y_train)
    # 模型评估
    y_predict=estimator.predict(x_test)
    print(y_predict)
    print("真实值和预测值：\n",y_test==y_predict)

    score=estimator.score(x_test,y_test)
    print("准确率：\n",score)

    return None

if __name__ == '__main__':
    # 使用朴素贝叶斯算法对新闻进行分类
    nb_news()

