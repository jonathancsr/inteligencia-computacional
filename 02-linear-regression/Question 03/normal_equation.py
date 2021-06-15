import pandas as pd
import numpy as np

data = pd.read_csv("data2.txt", names=['tamanho', 'quartos', 'preco'])
df = pd.DataFrame(data=data)

X = df[['tamanho', 'quartos']]
Y = df['preco']

x0 = np.ones(Y.values.size)
X.insert(0, 'x0', x0)

Y_matriz = Y.values
X_matriz = X.values
Xt_matriz = X.T.values
XtX = X.T.dot(X)
X_inverse = np.linalg.inv(XtX.values)

teta = X_inverse.dot(Xt_matriz).dot(Y_matriz)
coust = X.dot(teta) - Y

print(teta)
print("Custo: ", np.sum(coust ** 2) / (2 * len(coust)))
