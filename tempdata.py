from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from graphviz import Source
from sklearn.tree import export_graphviz

# для проверки классификатора:

df = pd.read_csv('iris.csv', index_col=0).reset_index(drop=True)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
y = df['target']
X = df.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mean = accuracy_score(y_test, y_pred)
print(mean)

export_graphviz(
                decision_tree=clf,
                out_file='tree_clf.dot',
                feature_names = X.columns,
                filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree_clf.dot', '-o', 'skl_clf_tree.png', '-Gdpi=600'])

# для проверки регрессии:

# df = pd.read_csv('boston.csv', index_col=0).reset_index(drop=True)
# y = df['target']
# X = df.drop(['target'], axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#
# reg = DecisionTreeRegressor(max_depth=4)
#
# reg.fit(X_train, y_train)
#
# y_pred = reg.predict(X_test)
# mean = mean_squared_error(y_test, y_pred)
# print(mean)
#
# export_graphviz(
#                 decision_tree=reg,
#                 out_file='tree.dot',
#                 feature_names = X.columns,
#                 filled = True)
#
# from subprocess import call
# call(['dot', '-Tpng', 'tree.dot', '-o', 'skl_reg_tree.png', '-Gdpi=600'])


