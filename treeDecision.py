class DecisionTree:

    @classmethod
    def predict(cls, root, x):
        cur_pos = root
        while True:
            i = cur_pos.indx
            if x[i] <= cur_pos.value:
                cur_pos = cur_pos.left
            else:
                cur_pos = cur_pos.right
            if not cur_pos.target is None:
                return cur_pos.target

    @classmethod
    def add_obj(cls, obj, node=None, left=True):
        if not node is None:
            if left:
                node.left = obj
            else:
                node.right =obj
        return obj
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
x1 = [7.0, 3.2, 4.7, 1.4] # target = 1
x2 = [5.8, 2.7, 5.1, 1.9] # target = 2
x3 = [6.2, 3.4, 5.4, 2.3] # target = 2
x4 = [6.7, 3.1, 4.7, 1.5] # target = 1

res = DecisionTree.predict(root, x2)
print(res)
