import numpy as np

class TreeObj:

    def __init__(self, indx=-1, split_value=None, ans=None, impurity=1.0, samples=0, value=None):
        self.indx = indx
        self.split_value = split_value
        self.ans = ans
        self.impurity = impurity
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
        self.cnt_classes = None

    def predict(self, X):
        y_predict = []
        for x in X:
            cur_pos = self.tree
            while cur_pos.ans is None:
                i = cur_pos.indx
                if x[i] <= cur_pos.split_value:
                    cur_pos = cur_pos.left
                else:
                    cur_pos = cur_pos.right
            y_predict.append(cur_pos.ans)
        return np.array(y_predict)

    @staticmethod
    def add_obj(obj, node=None, left=True):
        if not node is None:
            if left:
                node.left = obj
            else:
                node.right =obj
        return obj

    @staticmethod
    def accuracy(y_true, y_pred):
        try:
            result = (y_true == y_pred)
            return round(np.sum(result) / len(result), 2)
        except Exception as e:
            return "Error: ==== \n" + str(e)

    # только числа без пропусков

    def fit(self, X_train, y_train):

        def impurity(y_part, type='entropy'):
            if type == 'entropy':
                target_class, cnt_target = np.unique(y_part, return_counts=True)
                probable = cnt_target / len(y_part)
                entropy = np.inner(-probable, np.log2(probable))
                return round(float(entropy), 3)
        # чтоб понимать сколько каких классов попало в ветвь дерева
        def get_count_classes(y):
            return np.bincount(y, minlength=self.cnt_classes).tolist()

        def train(X, y, root=None, left=True, depth=self.max_depth):
            target_class = np.unique(y, return_counts=True)
            if depth == 0:
                ans = target_class[0][target_class[1] == target_class[1].max()][0]
                self.add_obj(TreeObj(ans=ans,
                                     impurity=0,
                                     samples=y.shape[0],
                                     value=get_count_classes(y)),
                             root, left)
            elif len(target_class[1]) <= 1:
                # один класс, фиксируем лист
                self.add_obj(TreeObj(ans=y[0],
                                     impurity=0,
                                     samples=y.shape[0],
                                     value=get_count_classes(y)), root, left)

            else:
                s_0 = impurity(y)
                information_gain_max = 0
                row_split, col_split = 0, 0
                for col in range(X.shape[1]):
                    feature = X[:, col]
                    for row in range(feature.shape[0]):
                        left_data = y[feature <= feature[row]]
                        right_data = y[feature > feature[row]]
                        impurity_left = impurity(left_data)
                        impurity_right= impurity(right_data)
                        ig = round(s_0 - (len(left_data) * impurity_left + len(right_data) * impurity_right)/(len(left_data) + len(right_data)), 3)
                        if ig > information_gain_max:
                            information_gain_max = ig
                            row_split, col_split = row, col
                node = self.add_obj(TreeObj(indx=col_split,
                                            split_value=X[row_split, col_split],
                                            impurity=s_0,
                                            samples=len(X),
                                            value=get_count_classes(y)),
                                    root, left)
                if root is None:
                    self.tree = node
                # рекурсия по веткам
                left_values = X[:, col_split] <= X[row_split, col_split]
                right_values = np.logical_not(left_values)
                train(X[left_values], y[left_values], root=node, depth=depth-1)
                train(X[right_values], y[right_values], root=node, left=False, depth=depth-1)

        X = np.array(X_train)
        y = np.array(y_train)
        self.cnt_classes = len(np.unique(y))
        train(X, y, depth=self.max_depth)

class DecisionTreeReg:

    def __init__(self, *args, max_depth=2):
        self.tree = None
        self.max_depth=max_depth

    def predict(self, X):
        y_predict = []
        for x in X:
            cur_pos = self.tree
            while cur_pos.ans is None:
                i = cur_pos.indx
                if x[i] <= cur_pos.split_value:
                    cur_pos = cur_pos.left
                else:
                    cur_pos = cur_pos.right
            y_predict.append(cur_pos.ans)
        return np.array(y_predict)

    @staticmethod
    def add_obj(obj, node=None, left=True):
        if not node is None:
            if left:
                node.left = obj
            else:
                node.right =obj
        return obj

    @staticmethod
    def mse(y_true, y_pred):
        try:
            return np.round(np.mean((y_true-y_pred) ** 2), 2)
        except:
            return "Error"

    # только числа без пропусков

    def fit(self, X_train, y_train):

        def impurity(y_part):
            avg = np.mean(y_part)
            return np.mean((y_part-avg) ** 2)

        def train(X, y, root=None, left=True, depth=self.max_depth):
                if depth == 0 or len(y) <= 2:
                    ans = np.round(np.mean(y), 3)
                    self.add_obj(TreeObj(ans=ans,
                                         impurity=np.round(impurity(y), 3),
                                         samples=y.shape[0]),
                                 root, left)
                else:
                    s_0 = impurity(y)
                    information_gain_max = 0
                    row_split, col_split = 0, 0
                    for col in range(X.shape[1]):
                        feature = X[:, col]
                        for row in range(feature.shape[0]):
                            left_data = y[feature <= feature[row]]
                            right_data = y[feature > feature[row]]
                            if len(left_data) <= 1 or len(right_data) <= 1:
                                continue
                            impurity_left = impurity(left_data)
                            impurity_right= impurity(right_data)
                            ig = round(s_0 - (len(left_data) * impurity_left + len(right_data) * impurity_right)/(len(left_data) + len(right_data)), 4)
                            if ig > information_gain_max:
                                information_gain_max = ig
                                col_split, row_split = col, row
                    if not information_gain_max == 0:

                        node = self.add_obj(TreeObj(indx=col_split,
                                                    split_value=X[row_split, col_split],
                                                    impurity=np.round(s_0, 3),
                                                    samples=X.shape[0]),
                                            root, left)
                        if root is None:
                            self.tree = node

                        # рекурсия по веткам
                        left_values = X[:, col_split] <= X[row_split, col_split]
                        right_values = np.logical_not(left_values)
                        train(X[left_values], y[left_values], root=node, depth=depth - 1)
                        train(X[right_values], y[right_values], root=node, left=False, depth=depth - 1)
                    else:
                        ans = np.round(np.mean(y), 3)
                        self.add_obj(TreeObj(ans=ans,
                                             impurity=np.round(impurity(y), 3),
                                             samples=y.shape[0]),
                                     root, left)

        X = np.array(X_train)
        y = np.array(y_train)
        train(X, y, depth=self.max_depth)
