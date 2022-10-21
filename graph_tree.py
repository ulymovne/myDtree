import graphviz

global_name_node = 0

def parse_tree(tree, feature_names, graph, parrent=None):
    global global_name_node
    global_name_node += 1
    if not tree.ans is None:
        name_node = global_name_node
        graph.node(str(name_node),
                   label=f"impurity = {tree.impurity}\nsamples = {tree.samples}\nvalue = {tree.value}\nanswer={tree.ans}",
                   fillcolor="green")
        if not parrent is None:
            graph.edge(str(parrent), str(name_node))
    else:
        name_node = global_name_node
        graph.node(str(name_node),
                   label=f"{feature_names[tree.indx]} <= {tree.split_value}\n\nimpurity = {tree.impurity}\nsamples = {tree.samples}\nvalue = {tree.value}",
                   fillcolor="#FFFFFF")
        if not parrent is None:
            graph.edge(str(parrent), str(name_node))
        parse_tree(tree.left, feature_names, graph, parrent=name_node)
        parse_tree(tree.right, feature_names, graph, parrent=name_node)
def get_graph(tree, feature_names, filename='tree.gv'):
    graph = graphviz.Digraph('Tree', filename=filename)
    graph.attr('node', shape='box', style="filled", color="black", fontname="helvetica")
    parse_tree(tree, feature_names, graph=graph)
    graph.view()