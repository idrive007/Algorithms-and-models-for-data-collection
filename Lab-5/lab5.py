#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def get_file_system_graph(root_dir):
    G = nx.Graph()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        G.add_node(dirpath, type='directory')
        # Додамо файли та каталоги як вузли та створимо ребра
        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            G.add_node(dir_full_path, type='directory')
            G.add_edge(dirpath, dir_full_path)
        for filename in filenames:
            file_full_path = os.path.join(dirpath, filename)
            G.add_node(file_full_path, type='file', extension=os.path.splitext(filename)[1].lower())
            G.add_edge(dirpath, file_full_path)
    return G

def set_node_colors(G):
    node_colors = {}
    for node in G.nodes:
        if G.nodes[node]['type'] == 'directory':
            sub_nodes = [n for n in G.neighbors(node)]
            sub_types = [G.nodes[sub_node]['type'] for sub_node in sub_nodes]
            if sub_types:
                most_common_type = Counter(sub_types).most_common(1)[0][0]
                node_colors[node] = 'blue' if most_common_type == 'directory' else 'green'
            else:
                node_colors[node] = 'blue'
        else:
            extension = G.nodes[node].get('extension', '')
            if extension in ['.txt', '.md']:
                node_colors[node] = 'yellow'
            elif extension in ['.py', '.ipynb']:
                node_colors[node] = 'red'
            elif extension in ['.png', '.jpg', '.jpeg']:
                node_colors[node] = 'orange'
            else:
                node_colors[node] = 'grey'
    return node_colors

def visualize_graph(G, node_colors):
    plt.figure(figsize=(20, 20))
    
    layout_functions = [
        (nx.kamada_kawai_layout, 'Kamada-Kawai'),
        (nx.circular_layout, 'Circular'),
        (nx.spectral_layout, 'Spectral'),
        (nx.spring_layout, 'Spring')
    ]
    
    for i, (layout_func, title) in enumerate(layout_functions, 1):
        plt.subplot(2, 2, i)
        pos = layout_func(G)
        color_values = [node_colors[node] for node in G.nodes]
        
        options = {
            'node_color': color_values,
            'node_size': 50,
            'width': 0.5,
            'with_labels': False,
            'alpha': 0.7
        }

        nx.draw(G, pos, **options)
        plt.title(title)

    legend_colors = {
        'Directory': 'blue',
        'File (Text)': 'yellow',
        'File (Code)': 'red',
        'File (Image)': 'orange',
        'Other File': 'grey'
    }
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                       for label, color in legend_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root_dir = r"D:/sublime"
    G = get_file_system_graph(root_dir)
    node_colors = set_node_colors(G)
    visualize_graph(G, node_colors)


# In[ ]:




