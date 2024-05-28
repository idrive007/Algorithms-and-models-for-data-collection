#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from instagrapi import Client
import networkx as nx
import matplotlib.pyplot as plt
from time import sleep

# Вхід до Інстаграму
cl = Client()
USERNAME = 'username'
PASSWORD = 'password'
cl.login(USERNAME, PASSWORD)

# фоловери
my_followings = cl.user_following(cl.user_id)
my_followings_names = [user.username for user in my_followings.values()]

# ініціалізація графу
G = nx.Graph()
G.add_node(cl.username, label=cl.username)

# додавання вузлів
for following in my_followings.values():
    G.add_node(following.username, label=following.full_name)
    G.add_edge(cl.username, following.username)

# фоловери моїх фоловерів
for person in my_followings.values():
    sleep(1)  
    try:
        following_followings = cl.user_following(person.pk)
        for following in following_followings.values():
            if following.username in my_followings_names:
                G.add_node(following.username, label=following.full_name)
                G.add_edge(person.username, following.username)
    except Exception as e:
        print(f"Error fetching data for {person.username}: {e}")

# збереження графу для gephi
nx.write_gexf(G, "InstaFriends.gexf")

# візуалізація графу за допопомогою networkx
print("Drawing...")
nx.draw_spring(G, with_labels=True, font_weight='bold', font_size=5)
plt.savefig('InstaGraf.png', dpi=600)
plt.show()

#  Отримання кількості edges у графі
num_edges = G.number_of_edges()
print(f"Number of edges in the graph: {num_edges}")

