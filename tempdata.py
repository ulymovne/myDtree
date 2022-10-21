from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from graphviz import Source
from sklearn.tree import export_graphviz

# db = load_iris(return_X_y=True, as_frame=True)
# X = db[0]
# y = db[1]
# db = X
# db['target'] = y
# db.to_csv('iris.csv')
#db.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']


#df = pd.read_csv('iris.csv', index_col=0).reset_index(drop=True)
df = pd.read_csv('boston.csv', index_col=0).reset_index(drop=True)
#sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='target')
#plt.show()
#df['target2'] = df['target'] == 1

y = df['target']
X = df.drop(['target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#X_train, y_train = X, y
# sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='target')
# plt.show()
#
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
reg = DecisionTreeRegressor(max_depth=6)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mean = mean_squared_error(y_test, y_pred)
print(mean)

export_graphviz(
                decision_tree=reg,
                out_file='tree.dot',
                feature_names = X.columns,
                filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


