#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pygame
import sys
import math

# Ініціалізація Pygame
pygame.init()

# Визначення кольорів
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Визначення розмірів вікна та прямокутника
WIDTH, HEIGHT = 800, 600
RECT_WIDTH, RECT_HEIGHT = 400, 300
CENTER = (WIDTH // 2, HEIGHT // 2)

# Створення вікна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Анімація стрілки")

# Функція, що малює стрілку
def draw_arrow(position):
    length = min(RECT_WIDTH, RECT_HEIGHT) // 2 - 50
    x1 = position[0]
    y1 = position[1]
    x2 = CENTER[0]
    y2 = CENTER[1]
    
    arrow_end_1 = (x1, y1)
    arrow_end_2 = (x2, y2)
    
    pygame.draw.line(screen, WHITE, arrow_end_1, arrow_end_2, 5)


# Основний цикл програми
def main():
    angle = 0
    position = (CENTER[0] - RECT_WIDTH // 2, CENTER[1] - RECT_HEIGHT // 2)
    perimeter = 2 * (RECT_WIDTH + RECT_HEIGHT)  # Периметр прямокутника
    speed = 2  # Швидкість руху по периметру

    clock = pygame.time.Clock()

    running = True
    while running:
        screen.fill(BLACK)

        # Обробка подій
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Обчислення позиції точки на прямокутнику
        angle += speed / perimeter
        if angle > 1:
            angle -= 1
        x = CENTER[0] - RECT_WIDTH // 2 + RECT_WIDTH * angle
        y = CENTER[1] - RECT_HEIGHT // 2
        if angle > 0.5:
            y += RECT_HEIGHT
            x = CENTER[0] + RECT_WIDTH // 2 - RECT_WIDTH * (angle - 0.5) * 2

        position = (x, y)

        # Малювання прямокутника
        pygame.draw.rect(screen, WHITE, (CENTER[0] - RECT_WIDTH // 2, CENTER[1] - RECT_HEIGHT // 2, RECT_WIDTH, RECT_HEIGHT), 2)

        # Малювання та оновлення екрану
        draw_arrow(position)
        pygame.display.flip()

        # Затримка для стабілізації частоти кадрів
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


# In[23]:


import pygame
import sys
import math

# Ініціалізація Pygame
pygame.init()

# Визначення кольорів
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Визначення розмірів вікна та прямокутника
WIDTH, HEIGHT = 800, 600
RECT_WIDTH, RECT_HEIGHT = 400, 300
CENTER = (WIDTH // 2, HEIGHT // 2)

# Створення вікна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Анімація стрілки")

# Функція, що малює стрілку
def draw_arrow(position):
    # Визначення точок для малювання стрілки
    arrow_points = [(position[0] - 15, position[1]),
                    (position[0] + 15, position[1]),
                    (position[0] + 15, position[1] - 10),
                    (position[0] + 30, position[1] - 10),
                    (position[0] + 30, position[1] - 20),
                    (position[0] + 40, position[1]),
                    (position[0] + 30, position[1] + 20),
                    (position[0] + 30, position[1] + 10),
                    (position[0] + 15, position[1] + 10),
                    (position[0] + 15, position[1])]
    # Малювання стрілки
    pygame.draw.polygon(screen, WHITE, arrow_points)

# Основний цикл програми
def main():
    angle = 0
    position = (CENTER[0] - RECT_WIDTH // 2, CENTER[1] - RECT_HEIGHT // 2)
    perimeter = 2 * (RECT_WIDTH + RECT_HEIGHT)  # Периметр прямокутника
    speed = 2  # Швидкість руху по периметру

    clock = pygame.time.Clock()

    running = True
    while running:
        screen.fill(BLACK)

        # Обробка подій
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Обчислення позиції точки на прямокутнику
        angle += speed / perimeter
        if angle > 1:
            angle -= 1
        x = CENTER[0] - RECT_WIDTH // 2 + RECT_WIDTH * angle
        y = CENTER[1] - RECT_HEIGHT // 2
        if angle > 0.5:
            y += RECT_HEIGHT
            x = CENTER[0] + RECT_WIDTH // 2 - RECT_WIDTH * (angle - 0.5) * 2

        position = (x, y)

        # Малювання прямокутника
        pygame.draw.rect(screen, WHITE, (CENTER[0] - RECT_WIDTH // 2, CENTER[1] - RECT_HEIGHT // 2, RECT_WIDTH, RECT_HEIGHT), 2)

        # Малювання та оновлення екрану
        draw_arrow(position)
        pygame.display.flip()

        # Затримка для стабілізації частоти кадрів
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


# In[ ]:




