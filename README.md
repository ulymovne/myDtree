<h2>Описание</h2>
Самописное дерево решений для классификации и регрессии.<br>
Был интерес разобраться с алгоритмом и закодить его.
Не акцентировался на всевозможных исключениях, частных случаях и тд. Просто голый алгоритм работы с числовыми признаками без пропусков.<br>

Чтобы удобно было смотреть, что строит дерево, сделал визуализацию через билиотеку graphviz. Реализовано файле graph_tree.py в виде функции.<br>

В основном модуле, для классификации использую данные ирисов, для регрессии данные по оценке стоимости недвижимости Бостона. Сплитую данные стандартным методом из sklearn.<br>

В модуле treeDecision есть три класса: непосредственно класс дерева TreeObj, класс классификации DecisionTreeCF, и класс регрессии DecisionTreeReg (почти копипаст классификации с небольшими изменениями).<br>

В tempdata.py код для проверки и сравнения, использую стандартную sklearn библиотеку. 



