from graphviz import Digraph

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = set(), []

    def build(v):
        if v not in nodes:
            nodes.add(v)

            label = f"{v.label} | data: {v.data:.4f}"
            dot.node(str(id(v)), label = "{ %s | data %.4f | grad %.4f }" % (v.label, v.data, v.grad), shape='record')

            for child in v._prev:
                edges.append((child, v))
                build(child)

    build(root)

    for u, v in edges:
        dot.edge(str(id(u)), str(id(v)), label=v._op)

    return dot
