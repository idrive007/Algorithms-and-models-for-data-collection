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

# In[8]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Функція для обчислення векторного поля
def vector_field(x, y, z):
    denominator = x**2 + y**2 + z**2
    F1 = y * z / denominator
    F2 = x * z / denominator
    F3 = x * y / denominator
    return F1, F2, F3

# Сітка точок
x = np.linspace(-3, 4, 10)
y = np.linspace(-3, 4, 10)
z = np.linspace(-3, 4, 10)
X, Y, Z = np.meshgrid(x, y, z)

# Обчислення компоненти векторного поля на сітці точок
U, V, W = vector_field(X, Y, Z)

# Візуалізація векторного поля
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Відображення векторного поля за допомогою quiver
ax.quiver(X, Y, Z, U, V, W, length=0.2, color='black')

# Налаштування відображення
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Vector Field Visualization')
plt.show()


# # 4. Побудуйте візуалізацію тензорного поля за допомогою еліпсоїдів, кубоїдів, циліндрів та будь-якого суперквадру.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Функція для візуалізації тензорного поля
def visualize_tensor_field(object_type):
    # Задаємо діапазон значень
    x = np.linspace(-2*np.pi, 2*np.pi, 5)
    y = np.linspace(-2*np.pi, 2*np.pi, 5)
    z = np.linspace(-2*np.pi, 2*np.pi, 5)
    
    # Створення сітки точок
    X, Y, Z = np.meshgrid(x, y, z)
    
    # компоненти тензора
    T11 = np.sin(X)
    T12 = X + Y
    T13 = X + Z
    T21 = np.cos(Y)
    T22 = Y + Z
    T31 = np.cos(Z)
    
    # Візуалізація
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Відображення тензорного поля за допомогою обраного типу об'єкта
    if object_type == 'ellipsoid':
        plot_ellipsoids(ax, X, Y, Z, T11, T12, T13)
    elif object_type == 'cuboid':
        plot_cuboids(ax, X, Y, Z, T11, T12, T13)
    elif object_type == 'cylinder':
        plot_cylinders(ax, X, Y, Z, T11, T12, T13)
    elif object_type == 'superquadric':
        plot_superquadrics(ax, X, Y, Z, T11, T12, T13)
    else:
        raise ValueError("Unknown object type. Use 'ellipsoid', 'cuboid', 'cylinder', or 'superquadric'.")
    
    # Налаштування відображення
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'Visualization of Tensor Field using {object_type.capitalize()}s')
    
    plt.show()

# Функції для створення різних об'єктів
def plot_ellipsoids(ax, X, Y, Z, T11, T12, T13):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = T11[i, j, k] * np.outer(np.cos(u), np.sin(v))
                y = T12[i, j, k] * np.outer(np.sin(u), np.sin(v))
                z = T13[i, j, k] * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x + X[i, j, k], y + Y[i, j, k], z + Z[i, j, k], color='b', alpha=0.5)

def plot_cuboids(ax, X, Y, Z, T11, T12, T13):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                x = X[i, j, k]
                y = Y[i, j, k]
                z = Z[i, j, k]
                dx = T11[i, j, k] * 0.1
                dy = T12[i, j, k] * 0.1
                dz = T13[i, j, k] * 0.1
                ax.bar3d(x, y, z, dx, dy, dz, color='r', alpha=0.5)

def plot_cylinders(ax, X, Y, Z, T11, T12, T13):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                theta = np.linspace(0, 2 * np.pi, 100)
                z = np.linspace(0, 1, 100)
                theta, z = np.meshgrid(theta, z)
                x = T11[i, j, k] * np.cos(theta)
                y = T12[i, j, k] * np.sin(theta)
                z = T13[i, j, k] * z
                ax.plot_surface(x + X[i, j, k], y + Y[i, j, k], z + Z[i, j, k], color='g', alpha=0.5)

def plot_superquadrics(ax, X, Y, Z, T11, T12, T13):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                eta = np.linspace(-np.pi / 2, np.pi / 2, 100)
                omega = np.linspace(-np.pi, np.pi, 100)
                eta, omega = np.meshgrid(eta, omega)
                x = T11[i, j, k] * np.cos(eta)**2 * np.cos(omega)**2
                y = T12[i, j, k] * np.cos(eta)**2 * np.sin(omega)**2
                z = T13[i, j, k] * np.sin(eta)**2
                ax.plot_surface(x + X[i, j, k], y + Y[i, j, k], z + Z[i, j, k], color='m', alpha=0.5)

visualize_tensor_field('ellipsoid')
visualize_tensor_field('cuboid')
visualize_tensor_field('cylinder')
visualize_tensor_field('superquadric')


# In[ ]:




