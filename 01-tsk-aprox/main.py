# Fuzzy TSK para a aproximacao X e Y:
import numpy as np
import matplotlib.pyplot as plt

x = []
y = []
w1 = []
w2 = []
z1 = []
z2 = []
betas = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

betasX = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

arq = open('entry.txt', 'r')
text = arq.readlines()
for line in text:
    x.append(float(line.split(' ')[0]))
    y.append(float(line.split(' ')[1].split('\n')[0]))
arq.close()

for i in x:
    w1.append(0.67 * i + 0.33)
    w2.append(-0.67 * i + 0.33)

i = 0
for i in range(12):
    betas[0][i] = (w1[i] / (w1[i] + w2[i]))
    betas[1][i] = ((w2[i] / w1[i] + w2[i]))
i = 0
for i in range(12):
    betasX[i][0] = betas[0][i]
    betasX[i][1] = betas[1][i]
    betasX[i][2] = betas[0][i] * x[i]
    betasX[i][3] = betas[1][i] * x[i]

betasXPseInv = np.linalg.pinv(betasX)
P = np.dot(betasXPseInv, y)
Z = np.dot(betasX, P)
print(betasX)

plt.plot(Z)
plt.show()
