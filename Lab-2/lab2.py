#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # 1. Візуалізацію скалярного поля. Знайдіть його градієнт та візуалізуйте його як плоске векторне поле;
# 

# ### Варіант 2
# 


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



# In[2]:


# # 2.	Побудуйте візуалізацію плоского векторного поля як за допомогою векторів та ліній току з бібліотеки matplotlib та за допомогою коду з лістингу.
# 


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



# In[3]:


# # 3. Побудуйте тривимірну візуалізацію векторного поля; За додатковий бал (не обов’язково) модернізуйте алгоритм побудови ліній току на випадок 3-вимірного поля.


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



# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import glyph_visualization_lib as gvl

def visualize_tensor_field(glyph_type, superquadrics_option=0, glyph_name=''):
   
    x = np.linspace(-2 * np.pi, 2 * np.pi, 8, dtype=float, endpoint=True)
    y = np.linspace(-2 * np.pi, 2 * np.pi, 8, dtype=float, endpoint=True)
    z = np.linspace(-2 * np.pi, 2 * np.pi, 8, dtype=float, endpoint=True)
    X, Y, Z = np.meshgrid(x, y, z)

   
    tensor_field = np.zeros((3, 3, X.shape[0], X.shape[1], X.shape[2]))
    tensor_field[0, 0] = np.sin(X)
    tensor_field[0, 1] = X + Y
    tensor_field[0, 2] = X + Z
    tensor_field[1, 1] = np.cos(Y)
    tensor_field[1, 2] = Y + Z
    tensor_field[2, 2] = np.cos(Z)

    
    vm_stress = gvl.get_von_Mises_stress(tensor_field)

  
    glyph_radius = 0.25
    limits = [np.min(vm_stress), np.max(vm_stress)]
    colormap = plt.get_cmap('rainbow', 120)
    fig = mlab.figure(bgcolor=(1, 1, 1))

  
    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                center = [x[i], y[j], z[k]]
                data = tensor_field[:, :, i, j, k]
                color = colormap(gvl.get_colormap_ratio_on_stress(vm_stress[i, j, k], limits))[:3]

             
                x_g, y_g, z_g = gvl.get_glyph_data(center, data, limits, glyph_points=12, glyph_radius=glyph_radius, glyph_type=glyph_type, superquadrics_option=superquadrics_option)
              
                mlab.mesh(x_g, y_g, z_g, color=color)

    mlab.move(forward=1.8)
    filename = f"tensor_field_visualization_{glyph_name}.png"
    mlab.savefig(filename, size=(1000, 1000))
    mlab.show()

if __name__ == '__main__':
    # Visualize using ellipsoids
    visualize_tensor_field(glyph_type=2, glyph_name='ellipsoids')
    # Visualize using cuboids
    visualize_tensor_field(glyph_type=0, glyph_name='cuboids')
    # Visualize using cylinders
    visualize_tensor_field(glyph_type=1, glyph_name='cylinders')
    # Visualize using superquadrics
    visualize_tensor_field(glyph_type=3, superquadrics_option=2, glyph_name='superquadrics')


# In[ ]:




