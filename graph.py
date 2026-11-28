from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    
    def build(value):
        if value not in nodes:
            nodes.add(value)
            for child in value._prev:
                edges.add((child, value))
                build(child)
        
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    
    nodes, edges = trace(root)
    
    for node in nodes:
        uid = str(id(node))
        
        dot.node(name = uid, label = f"{{ {node.label} | data {node.data:.4f} | gradient {node.gradient:.4f}}}", shape = "record")
        if node._op:
            dot.node(name = uid + node._op, label = node._op)
            dot.edge(uid + node._op, uid)
            
            
    for node1, node2 in edges:
        dot.edge(str(id(node1)), str(id(node2)) + node2._op)
        
    return dot
        