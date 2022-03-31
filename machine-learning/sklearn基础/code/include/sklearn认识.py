from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd
import matplotlib.pyplot as plt

def dataset_demo():
    '''
    sklearn数据集使用
    :return:
    '''
    # 获取数据及
    iris=load_iris()
    print("鸢尾花数据集：\n",iris)
    print("查看数据集描述:\n",iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)


    # 数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值：\n",x_train,x_train.shape)
    return None

def dict_demo():
    '''
    字典值特征抽取
    :return:
    '''
    data=[{'city': '北京','temperature':100},{'city': '上海','temperature':60},{'city': '深圳','temperature':30}]

    # 实例化一个转换器类
    transfer=DictVectorizer(sparse=False)
    # 调用fit_transferrom()
    data_new=transfer.fit_transform(data)

    print("data_new:",data_new)
    # print("特征名字：\n",transfer.feature_names_)
    print("特征名字：\n",transfer.get_feature_names())

    return None


def count_demo():
    '''
    文本特征抽取 ： CountVecotrizer
    :return:
    '''
    data=["life is short,i like python","life is too long,i dislike python"]

    # 实例化转换器类
    transfer=CountVectorizer(stop_words=["is","too"])
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n",transfer.get_feature_names())

    return None



def count_chinese_demo():
    '''
    中文文本特征抽取 ： CountVecotrizer
    :return:
    '''
    data=["现代 汉语 方言 一般 可分为 官话 方言 吴 方言 湘 方言 客家 方言 闽 方言 粤 方言 赣 方言 等"]

    # 实例化转换器类
    transfer=CountVectorizer()
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray())
    print("特征名字：\n",transfer.get_feature_names())

    return None


def cut_word(text):
    '''
    功能函数：用于专门中文分词
    :param text:
    :return:
    '''
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    '''
    中文文本特征提取--自动分词
    :return:
    '''
    data= ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = CountVectorizer()
    data_ret = transfer.fit_transform(data_new)
    print("data_new:\n", data_ret.toarray())
    print("特征名字：\n", transfer.get_feature_names())
    return None


def ifidf_demo():
    '''
    使用TF-IDF的方法进行文本特征抽取
    :return:
    '''
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = TfidfVectorizer(stop_words=["一种","所以"])
    data_ret = transfer.fit_transform(data_new)
    print("data_new:\n", data_ret.toarray())
    print("特征名字：\n", transfer.get_feature_names())

    return None

def minmax_demo():
    '''
    归一化
    :return:
    '''
    # 获取数据
    data=pd.read_csv("dating.txt",sep='\t')
    data=data.iloc[:,:3]
    # print(data)

    # 实例化一个转换器类
    transfer=MinMaxScaler(feature_range=(2,3))

    data_new=transfer.fit_transform(data)
    print(data_new)


    return  None

def stand_demo():
    '''
    标准化
    :return:
    '''
    # 获取数据
    data=pd.read_csv("dating.txt",sep='\t')
    data=data.iloc[:,:3]
    # print(data)

    # 实例化一个转换器类
    transfer=StandardScaler()

    data_new=transfer.fit_transform(data)
    print(data_new)


    return  None


def variance_demo():
    '''
    过滤低方差特征
    :return:
    '''
    data=pd.read_csv("factor_returns.csv")

    data_new=data.iloc[:,1:2]
    transfer=VarianceThreshold()
    data_ret=transfer.fit_transform(data_new)

    print(data_ret,data_ret.shape)

    # 计算某两个变量之间的相关系数
    print("data:", data)
    r=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("相关系数",r)

    plt.figure(figsize=(20,8),dpi=80)
    plt.scatter(data["revenue"],data['total_expense'])
    plt.show()

    return None


def pca_demo():
    '''
    PCA降维
    :return:
    '''
    data= [[2,8,4,5], [6,3,0,8], [5,4,9,1]]
    # 实例化一个转换器
    transfer=PCA(n_components=0.95)
    # 调用fit_transform
    data_new=transfer.fit_transform(data)

    print(data_new)
    return None



if __name__ == '__main__':
    # sklearn数据集的使用
    # dataset_demo()

    # 字典值特征抽取
    # dict_demo()

    # 文本特征抽取
    # count_demo()

    # 中文文本特征抽取
    # count_chinese_demo()

    #中文文本特征提取--自动分词
    # count_chinese_demo2()

    #使用TF-IDF的方法进行文本特征抽取
    # ifidf_demo()

    # 归一化
    # minmax_demo()

    # 标准化
    # stand_demo()

    # 过滤低方差特征
    # variance_demo()

    # PCA降维
    pca_demo()
