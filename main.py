from treeDecision import DecisionTreeCF, DecisionTreeReg
from sklearn.model_selection import train_test_split
from graph_tree import get_graph
import numpy as np

def test_clf():
    X = []
    y = []
    #тест на Ирисах
    with open('iris.csv') as file:
        file.readline()
        for data in file.readlines():
            str_ = data.strip().split(',')
            X.append(list(map(float, str_[1:5])))
            y.append(int(str_[5]))

    clf = DecisionTreeCF(max_depth=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    clf.fit(X_train, y_train)
    # посмотрим на дерево:
    get_graph(clf.tree, filename='my_tree_clf.gv', feature_names=['sepal_length','sepal_width','petal_length','petal_width'])

    res = clf.predict(X_test)
    print('accuracy_score = ', clf.accuracy(y_test, res))

def test_regres():
    X = []
    y = []
    with open('boston.csv') as file:
        file.readline()
        for data in file.readlines():
            str_ = data.strip().split(',')
            X.append(list(map(float, str_[1:14])))
            y.append(float(str_[14]))

    clf = DecisionTreeReg(max_depth=4)
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    clf.fit(X_train, y_train)

    get_graph(clf.tree, filename='my_tree_reg.gv', feature_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS' , 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'])
    res = clf.predict(X_test)
    print("mse_score = ", clf.mse(y_test, res))

if __name__=='__main__':
    test_clf()
    test_regres()