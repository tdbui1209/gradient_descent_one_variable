import matplotlib.pyplot as plt
import time
import numpy as np

#--------------Khoi tao data-----------------
X = np.random.rand(1000, 1)
noise = np.random.rand(1000, 1)
y = X * 20 + 20 + noise


w1 = 0
w0 = 0

#---------Truc quan hoa qua trinh train------
plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
plt.scatter(X, y, s=0.1)
line, = ax.plot(X, X*w1+w0, c='r')

text_1 = ax.text(0, 40, 'Epoch:')

ax.text(0, 39, 'w0   :')
ax.text(0, 38, 'w1   :')
ax.text(0, 37, 'J    :')

w1_text = ax.text(0.1, 39, '0')
w0_text = ax.text(0.1, 38, '0')
J_text = ax.text(0.1, 37, '0')

text = ax.text(0.1, 40, '0')

#--------------Training Model----------------
ites = 10000
learning_rate = 0.0001
n = X.shape[0]

fx = []
for i in range(ites):
    r1 = sum(X*(w0 + w1*X - y))
    r0 = sum(w0 + w1*X - y)
    w1 -= r1*learning_rate
    w0 -= r0*learning_rate
    x0 = np.linspace(0, 1, 2, endpoint=True)
    y0 = w0 + w1*x0
    J = (1/2) * (1/n) * sum(((X*w1+w0)-y)**2)
    fx.append(J)
    
    line.set_xdata(x0)
    line.set_ydata(y0)
    
    text.set_text(i)
    
    w1_text.set_text(float(w1))
    w0_text.set_text(float(w0))
    J_text.set_text(J)
    
    figure.canvas.draw()
    figure.canvas.flush_events()
    
    time.sleep(0) # Toc do hien thi

figure_2 = plt.figure()
plt.plot(fx)
