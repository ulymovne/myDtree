from graph_tree import get_graph
import numpy as np

class TreeObj:

    def __init__(self, indx=-1, split_value=None, ans=None, entropy=1, samples=0, value=None):
        self.indx = indx
        self.split_value = split_value
        self.ans = ans
        self.entropy = entropy
        self.samples = samples
        self.value = value
        self.__left = None
        self.__right = None


    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, branch):
        self.__left = branch

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, branch):
        self.__right = branch

class DecisionTreeCF:

    def __init__(self, *args, max_depth=2):
        self.tree = None
        self.max_depth=max_depth
        self.all_classes = None

    @classmethod
    def predict(cls, root, X):
        # пока без numpy
        y_predict = []
        for x in X:
            cur_pos = root
            while cur_pos.ans is None:
                i = cur_pos.indx
                if x[i] <= cur_pos.split_value:
                    cur_pos = cur_pos.left
                else:
                    cur_pos = cur_pos.right
            y_predict.append(cur_pos.ans)
        return y_predict
    @classmethod
    def add_obj(cls, obj, node=None, left=True):
        if not node is None:
            if left:
                node.left = obj
            else:
                node.right =obj
        return obj

    @staticmethod
    def accuracy(y_true, y_pred):
        try:
            result = [y_true[i] == y_pred[i] for i in range(len(y_true))]
            return round(sum(result) / len(result), 2)
        except:
            return "Error"

    # только числа без пропусков

    def fit(self, X_train, y_train):

        def entropy_(y_part):
            target_class = np.unique(y_part)
            ent = 0
            for i in target_class:
                p = len(y_part[y_part == i]) / len(y_part)
                ent += -p * np.log2(p)
            return round(ent, 3)
        # чтоб понимать сколько каких классов попало в ветвь дерева
        def get_count_classes(y):
            y_classes = np.unique(y, return_counts=True)
            return [y_classes[1][y_classes[0] == i][0] if i in y_classes[0] else 0 for i in self.all_classes]

        def train(X, y, root=None, left=True, depth=2):

            target_class = np.unique(y, return_counts=True)
            # пока сплитуем до победного
            if depth == 0:
                ans = target_class[0][target_class[1] == target_class[1].max()][0]
                self.add_obj(TreeObj(ans=ans,
                                     entropy=0,
                                     samples=y.shape[0],
                                     value=get_count_classes(y)),
                             root, left)
            elif target_class[1].shape[0] <= 1:
                # один класс, фиксируем лист
                self.add_obj(TreeObj(ans=y[0],
                                     entropy=0,
                                     samples=y.shape[0],
                                     value=get_count_classes(y)), root, left)

            else:
                s_0 = entropy_(y)
                information_gain_max = 0
                split_params = ()
                for x in range(X.shape[1]):
                    feature = X[:, x]
                    for value_index in range(feature.shape[0]):
                        left_data = y[feature <= feature[value_index]]
                        right_data = y[feature > feature[value_index]]
                        entropy_left = entropy_(left_data)
                        entropy_right= entropy_(right_data)
                        ig = round(s_0 - (len(left_data) * entropy_left + len(right_data) * entropy_right)/(len(left_data) + len(right_data)), 3)
                        if ig > information_gain_max:
                            information_gain_max = ig
                            split_params = (x, value_index, len(left_data))
                node = self.add_obj(TreeObj(indx=split_params[0],
                                            split_value=X[split_params[1], split_params[0]],
                                            entropy=s_0,
                                            samples=X.shape[0],
                                            value=get_count_classes(y)),
                                    root, left)
                if root is None:
                    self.tree = node

                # рекурсия по веткам
                index1 = X[:, split_params[0]] <= X[split_params[1], split_params[0]]
                index2 = X[:, split_params[0]] > X[split_params[1], split_params[0]]
                train(X[index1], y[index1], root=node, depth=depth-1)
                train(X[index2], y[index2], root=node, left=False, depth=depth-1)

        X = np.array(X_train)
        y = np.array(y_train)
        self.all_classes = np.unique(y)
        train(X, y, depth=self.max_depth)


## берем данные из файла iris2.csv
X = []
y = []
with open('iris.csv') as file:
    file.readline()
    for data in file.readlines():
        str_ = data.strip().split(',')
        X.append(list(map(float, str_[1:5])))
        y.append(int(str_[5]))
clf = DecisionTreeCF(max_depth=3)

clf.fit(X, y)
get_graph(clf.tree, ['sepal_length','sepal_width','petal_length','petal_width'])
res = DecisionTreeCF.predict(clf.tree, X)

print(DecisionTreeCF.accuracy(y, res))