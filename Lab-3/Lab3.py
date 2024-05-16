#!/usr/bin/env python
# coding: utf-8

# # 1. Візуалізацію скалярного поля. Знайдіть його градієнт та візуалізуйте його як плоске векторне поле;
# 

# ### Варіант 2
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Задання області
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)
X, Y = np.meshgrid(x, y)

# Скалярне поле
Z = X * np.sqrt(Y) + Y * np.sqrt(X)

# Візуалізація скалярного поля
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.title('Скалярне поле u(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Обчислення градієнту
grad_x = np.sqrt(Y) + 0.5 * X / np.sqrt(Y)
grad_y = np.sqrt(X) + 0.5 * Y / np.sqrt(X)

# Візуалізація плоского векторного поля
plt.figure(figsize=(10, 6))
plt.quiver(X, Y, grad_x, grad_y, scale=20)
plt.title('Градієнт скалярного поля')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# # 2.	Побудуйте візуалізацію плоского векторного поля як за допомогою векторів та ліній току з бібліотеки matplotlib та за допомогою коду з лістингу.
# 

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def u(x, y):
    return x ** 2 + 2 * y

def v(x, y):
    return y ** 2 + 2 * x

xx, yy = np.meshgrid(np.linspace(-4, 4, 10),
                     np.linspace(-4, 4, 10))

u_val = u(xx, yy)
v_val = v(xx, yy)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.quiver(xx, yy, u_val, v_val)
plt.title('Векторне поле')

plt.subplot(1, 2, 2)
plt.streamplot(xx, yy, u_val, v_val)
plt.title('Лінії току')

plt.tight_layout()
plt.show()


# # 3. Побудуйте тривимірну візуалізацію векторного поля; За додатковий бал (не обов’язково) модернізуйте алгоритм побудови ліній току на випадок 3-вимірного поля.

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Обчислення компонент векторного поля
def F(x, y, z):
    r_squared = x**2 + y**2 + z**2
    return y*z/r_squared, x*z/r_squared, x*y/r_squared

# Генерування значень для x, y, z
x = np.linspace(-3, 4, 10)
y = np.linspace(-3, 4, 10)
z = np.linspace(-3, 4, 10)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Обчислення компонент векторного поля на генерованій сітці
u, v, w = F(xx, yy, zz)

# Побудова візуалізації
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Побудова векторного поля
ax.quiver(xx, yy, zz, u, v, w, length=0.2, normalize=True, color='b')

# Модернізований алгоритм побудови ліній току
start_points = np.stack([np.random.choice(coord.flatten(), 10) for coord in [xx, yy, zz]], axis=-1)
stream = ax.streamplot(xx[:,0,0], yy[0,:,0], zz[0,0,:], u, v, w, 2, start_points=start_points, color='r')

# Налаштування відображення
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Тривимірне векторне поле та лінії току')
plt.show()


# # 4. Побудуйте візуалізацію тензорного поля за допомогою еліпсоїдів, кубоїдів, циліндрів та будь-якого суперквадру.

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Обчислюємо значення тензорного поля
x_values = np.linspace(-2*np.pi, 2*np.pi, 100)
y_values = np.linspace(-2*np.pi, 2*np.pi, 100)
z_values = np.linspace(-2*np.pi, 2*np.pi, 100)
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Задання тензорного поля
tensor_field = np.array([
    [np.sin(X), X + Y, X + Z],
    [X+Y, np.cos(Y), Y + Z],
    [X+Y, Y+Z, np.cos(Z)]
])

# Функція для побудови тензорного поля з різними геометричними фігурами
def plot_tensor_field(geometry, tensor_field):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(tensor_field.shape[0]):
        for j in range(tensor_field.shape[1]):
            for k in range(tensor_field.shape[2]):
                if not np.all(np.isnan(tensor_field[i, j, k])):
                    if geometry == 'ellipsoid':
                        ax.scatter(X[i, j, k], Y[i, j, k], Z[i, j, k], color='b', alpha=0.5)
                    elif geometry == 'cuboid':
                        ax.scatter(X[i, j, k], Y[i, j, k], Z[i, j, k], color='r', alpha=0.5, marker='s')
                    elif geometry == 'cylinder':
                        ax.scatter(X[i, j, k], Y[i, j, k], Z[i, j, k], color='g', alpha=0.5, marker='|')
                    elif geometry == 'superquadric':
                        ax.scatter(X[i, j, k], Y[i, j, k], Z[i, j, k], color='m', alpha=0.5, marker='^')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Tensor Field with {geometry}s')

# Побудова тензорного поля з різними геометричними фігурами
plot_tensor_field('ellipsoid', tensor_field)
plot_tensor_field('cuboid', tensor_field)
plot_tensor_field('cylinder', tensor_field)
plot_tensor_field('superquadric', tensor_field)

plt.show()


# In[ ]:




