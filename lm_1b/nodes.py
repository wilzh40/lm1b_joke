from gvgen import GvGen

class Node(object):
    def __init__(self, prefix):
        # { word: (probability, node)}
        self.prefix = prefix
        self.children  = {}
    

def visualize( t, fd=None ) :
    graph = GvGen()
    def walk( node, parent=None ) :
        if parent is None :
            parent = graph.newItem(t.prefix)
        for k, (probability, child_node) in node.children.items() :
            n = graph.newItem(k)
            link = graph.newLink(parent, n)
            graph.propertyAppend(link, "label", probability)
            walk(child_node, n)
    walk(t)
    if fd is None:
        return graph.dot()
    else: 
        return graph.dot(fd)


