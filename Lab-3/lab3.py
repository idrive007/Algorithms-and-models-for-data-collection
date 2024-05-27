#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# визначаємо прямокутник
rect = patches.Rectangle((2, 2), 6, 6, fill=False)

# Створюємо фігуру та вісь
fig, ax = plt.subplots()
ax.add_patch(rect)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal', adjustable='box')

# Визначаємо центр вікна
center = (5, 5)

#  Ініціалізуємо стрілку як None
arrow = patches.FancyArrowPatch(center, (2, 2), color='lavender', mutation_scale=20)
ax.add_patch(arrow)

# Функція для ініціалізації анімації
def init():
    return arrow,

#  Функція оновлення положення стрілки для кожного кадру
def animate(i):
    t = i / 100.0  # параметр від 0 до 1
    if t < 0.25:
        x = 2 + 8 * t * 4
        y = 2
    elif t < 0.5:
        x = 10
        y = 2 + 8 * (t - 0.25) * 4
    elif t < 0.75:
        x = 10 - 8 * (t - 0.5) * 4
        y = 10
    else:
        x = 2
        y = 10 - 8 * (t - 0.75) * 4
    
    # Оновлення положення стрілки
    arrow.set_positions(center, (x, y))
    return arrow,

# Створення анімації
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)

# збереження
anim.save('animation.gif', writer='pillow', fps=30)

plt.show()


# In[ ]:




