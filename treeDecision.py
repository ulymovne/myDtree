class DecisionTree:

    @classmethod
    def predict(cls, root, X):
        # пока без numpy
        y_predict = []
        for x in X:
            cur_pos = root
            while cur_pos.target is None:
                i = cur_pos.indx
                if x[i] <= cur_pos.value:
                    cur_pos = cur_pos.left
                else:
                    cur_pos = cur_pos.right
            y_predict.append(cur_pos.target)
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


class TreeObj:

    def __init__(self, indx, value=None, target=None):
        self.indx = indx
        self.value = value
        self.target = target
        self.__left = None
        self.__right = None

    @property
    def left(self):
        return self.__left

    @left.setter
    def left(self, param):
        self.__left = param

    @property
    def right(self):
        return self.__right

    @right.setter
    def right(self, param):
        self.__right = param

root = DecisionTree.add_obj(TreeObj(3, 1.75))

v_11 = DecisionTree.add_obj(TreeObj(2, 5.3), root)

v_111 = DecisionTree.add_obj(TreeObj(0, 4.95), v_11)
DecisionTree.add_obj(TreeObj(-1, None, 2), v_11, False)

v_1111 = DecisionTree.add_obj(TreeObj(3, 1.35), v_111)
DecisionTree.add_obj(TreeObj(-1, None, 1), v_111, False)

DecisionTree.add_obj(TreeObj(-1, None, 1), v_1111)
DecisionTree.add_obj(TreeObj(-1, None, 2), v_1111, False)

v_12 = DecisionTree.add_obj(TreeObj(2, 4.85), root, False)

v_121 = DecisionTree.add_obj(TreeObj(1, 3.1), v_12)
DecisionTree.add_obj(TreeObj(-1, None, 2), v_12, False)

DecisionTree.add_obj(TreeObj(-1, None, 2), v_121)
DecisionTree.add_obj(TreeObj(-1, None, 1), v_121, False)

# берем данные из файла iris2.csv
X = []
y = []
with open('iris2.csv') as file:
    file.readline()
    for data in file.readlines():
        str_ = data.strip().split(',')
        X.append(list(map(float, str_[1:5])))
        y.append(int(str_[5]))
res = DecisionTree.predict(root, X)

print(DecisionTree.accuracy(y, res))