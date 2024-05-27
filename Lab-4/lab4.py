#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import matplotlib.pyplot as plt
import networkx as nx

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

class Tree:
    def __init__(self):
        self.root = None

    def add_node(self, key, node=None):
        if node is None:
            node = self.root
        if self.root is None:
            self.root = Node(key)
        else:
            if key <= node.key:
                if node.left is None:
                    node.left = Node(key)
                    node.left.parent = node
                else:
                    self.add_node(key, node=node.left)
            else:
                if node.right is None:
                    node.right = Node(key)
                    node.right.parent = node
                else:
                    self.add_node(key, node=node.right)

    def tree_data(self, node=None):
        if node is None:
            node = self.root
        stack = []
        while stack or node:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node.key
                node = node.right

def add_edges(graph, node, pos, x=0, y=0, layer=1):
    if node is not None:
        pos[node.key] = (x, y)
        if node.left:
            graph.add_edge(node.key, node.left.key)
            l = x - 1 / layer
            add_edges(graph, node.left, pos, x=l, y=y-1, layer=layer+1)
        if node.right:
            graph.add_edge(node.key, node.right.key)
            r = x + 1 / layer
            add_edges(graph, node.right, pos, x=r, y=y-1, layer=layer+1)

def plot_tree(tree):
    graph = nx.DiGraph()
    pos = {}
    add_edges(graph, tree.root, pos)
    nx.draw(graph, pos, with_labels=True, arrows=False)
    plt.show()

t = Tree()
for _ in range(15):
    t.add_node(random.randint(-50, 50))

plot_tree(t)

    


# In[ ]:




