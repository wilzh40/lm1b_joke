from gvgen import GvGen

class Node(object):
    def __init__(self, prefix):
        # { word: (probability, node)}
        self.prefix = prefix
        self.children  = {}
    

def visualize( t, fd=None ) :
    graph = GvGen()
    def walk( node, total_prob, parent=None ) :
        if parent is None :
            parent = graph.newItem(t.prefix)

        for k, (prob, child_node) in node.children.items() :
            n = graph.newItem(k)
            link = graph.newLink(parent, n)
            graph.propertyAppend(link, "label", prob)
            walk(child_node, total_prob * prob, n)
    walk(t, 1)
    if fd is None:
        return graph.dot()
    else: 
        return graph.dot(fd)


