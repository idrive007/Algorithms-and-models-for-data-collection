#!/usr/bin/env python
# coding: utf-8

# # Варіант 11

# ### 1. Побудувати графіки функцій, поверхонь та стовпчикові діаграми. На всіх графіках підписати осі, відобразити сітку, легенду. Вивести текстом рівняння графіку функції.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# значення x
x = np.linspace(-5, 5, 400)

# Функції 
def y_func(x):
    return (4 + x**2 * np.exp(-3*x)) / (4 + np.sqrt(x**4 + np.sin(x)**2))

def z_func(x):
    return np.where(x <= 0, np.sqrt(1 + 5*x**2 - np.sin(x)**2), ((7 + x)**2) / np.cbrt(4 + np.exp(-0.7*x)))

# Побудова графіків
plt.figure(figsize=(14, 10))

# Графік функції Y(x)
plt.subplot(2, 1, 1)
plt.plot(x, y_func(x), label=r'$y=\frac{4+x^2e^{-3x}}{4+\sqrt{x^4+\sin^2(x)}}$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Графік функції Z(x)
plt.subplot(2, 1, 2)
plt.plot(x, z_func(x))
plt.xlabel('x')
plt.ylabel('z')

plt.text(-4, 5, r'$\sqrt{1+5x^2-\sin^2(x)}$, $x\leq0$', fontsize=12)
plt.text(2, 50, r'$\frac{(7+x)^2}{\sqrt[3]{4+e^{-0.7x}}}$, $x>0$', fontsize=12)

plt.grid(True)

plt.tight_layout()
plt.show()


# # Варіант 11

# ### 2. Побудувати поверхні

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# значення x та y
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)


def z_func(x, y):
    z = np.where(np.abs(x) + np.abs(y) < 0.5, x - np.exp(2*y),
                 np.where(np.abs(x) + np.abs(y) < 1, 2*x**2 - np.exp(y),
                          np.exp(5*x - 3) - y))
    return z


fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, z_func(X, Y), cmap='viridis')

# Підписи 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.grid(True)

equation_text = r'z = x - e^(2y),  |x| + |y| < 0.5\n'                 r'z = 2x^2 - e^y,   0.5 <= |x| + |y| < 1\n'                 r'z = e^(5x - 3) - y, |x| + |y| >= 1'
ax.text2D(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=12)

plt.show()


# # Варіант 11

# ### 3. Побудувати графіки у полярних координатах

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# параметр a
a = 1

# значення кута t від 0 до 2*pi
t = np.linspace(0, 2*np.pi, 1000)


x = a * np.cos(t)**3
y = a * np.sin(t)**3


plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.title('Астроїда: $x = a \cos^3(t)$, $y = a \sin^3(t)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()


# # Варіант 2 

# ### 4. Побудувати поверхні 2-го порядку. a, b, c – константи

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# константи a, b, c
a = 2
b = 3
c = 1

x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)

X, Y = np.meshgrid(x, y)

Z = np.sqrt((X**2 / a**2) + (Y**2 / b**2) - 1) * c

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('Однополосний гіперболоїд')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


# # Варіант 1

# ### 5. За даними з таблиць побудувати 2d та 3d стовпчикові діаграми

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Дані з таблиці
years = [1900, 1913, 1929, 1938, 1950, 1960, 1970, 1980, 1990, 2000]
countries = ['США', 'Німеччина', 'Франція', 'Японія', 'СРСР']
values = [
    [76.4, 97.6, 122.2, 130.5, 153, 176, 200.5, 227, 247, 277],
    [45.7, 54.7, 58.7, 62.3, 67, 72, 77, 78.5, 79, 82],
    [40.8, 41.8, 42, 42, 42, 46, 50.5, 54, 56.5, 59],
    [44, 51.6, 63.2, 71.8, 83, 93, 104, 116.8, 123.5, 127],
    [123, 158, 171.5, 186.5, 205.5, 226.5, 247, 258.5, 290, 290]
]


plt.figure(figsize=(10, 6))
for i, country in enumerate(countries):
    plt.plot(years, values[i], label=country)
plt.title('Зростання ВВП за роками')
plt.xlabel('Рік')
plt.ylabel('ВВП')
plt.legend()
plt.grid(True)
plt.xticks(years)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, country in enumerate(countries):
    ax.bar(years, values[i], zs=i, zdir='y', label=country)

ax.set_xlabel('Рік')
ax.set_ylabel('Країна')
ax.set_zlabel('ВВП')
ax.set_yticks(np.arange(len(countries)))
ax.set_yticklabels(countries)

ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.15))

plt.title('3D стовпчикова діаграма зростання ВВП за роками')
plt.tight_layout()
plt.show()


# In[ ]:




