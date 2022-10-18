import graphviz

global_name_node = 0

def parse_tree(tree, feature_names, graph, parrent=None):
    global global_name_node
    global_name_node += 1
    if not tree.target is None:
        name_node = global_name_node
        graph.node(str(name_node),
                   label=f"entropy = {tree.value}\nsamples = __\nvalue = [__, __]",
                   fillcolor="green")
        if not parrent is None:
            graph.edge(str(parrent), str(name_node))
    else:
        name_node = global_name_node
        graph.node(str(name_node),
                   label=f"{feature_names[tree.indx]} <= {tree.value}\n\nentropy = __\nsamples = __\nvalue = [__, __]",
                   fillcolor="#FFFFFF")
        if not parrent is None:
            graph.edge(str(parrent), str(name_node))
        parse_tree(tree.left, feature_names, graph, parrent=name_node)
        parse_tree(tree.right, feature_names, graph, parrent=name_node)
def get_graph(tree, feature_names):
    graph = graphviz.Digraph('Tree', filename='tree.gv')
    graph.attr('node', shape='box', style="filled", color="black", fontname="helvetica")
    parse_tree(tree, feature_names, graph=graph)
    graph.view()