import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Adaline:
    def __init__(self,learning_rate , n_iters):
        self.c = learning_rate
        self.n_iters = n_iters #number of iterations
        self.w = np.ones(3)
        self.activation_func = self.sigmfunc
        self.x_t = np.ones(3)
        self.s_error = []
        self.Em = np.array([])
    def sigmfunc(self,x):
        return 1 / (1 + np.exp(-x))

    def sigturev(self,x):
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

    def iter(self,x):
        x_train = np.ones(3)
        for idx,x_i in enumerate(x):
            x_train[0] = x_i[0]
            x_train[1] = x_i[1] *np.pi / 2
            w_t = np.transpose(self.w)
            v = np.dot(w_t,x_train)
            y = self.activation_func(v)
            yd = (3 * x_train[0] + 2 * np.cos(x_train[1]))/5
            e = yd-y

            self.w += self.c * e * self.sigturev(v) * x_train
            self.s_error.append(abs(e))

    def test(self,X):
        x_test = np.ones(3)
        Et = 0

        for idx, x_i in enumerate(X):
            x_test[0] = x_i[0]
            x_test[1] = x_i[1] *np.pi / 2

            w_t = np.transpose(self.w)
            v = np.dot(w_t, x_test)
            y = self.activation_func(v)
            yd = (3 * x_test[0] + 2 * np.cos(x_test[1])) / 5
            Et += abs(yd-y)

        return Et

train_set = np.random.rand(1000,2) #1000 elemanlı egitecek küme
data_set = np.random.rand(60,2) # 60 elemanlı denenecek küme


p = Adaline(learning_rate=1.2, n_iters=100)
p.iter(train_set)
Et = p.test(data_set)
Et = Et / len(data_set)

print("Ortalama hata:",Et)
print("Son w: ",p.w)
wd=-train_set[1][0]*p.w[0]-train_set[1][1]*p.w[1]-(3 * train_set[0] + 2 * np.cos(train_set[1]))*p.w[2]
x1, y1 = np.meshgrid(range(0,2), range(0,2))
z1 = (-p.w[0] * x1 - p.w[1] * y1 - wd) * 1. /p.w[2]
fig2=plt.figure()

ax= fig.add_subplot(111, projection='3d')
ax.scatter(train_set[:,0], train_set[:,1], (3 * train_set[:,0] + 2 * np.cos(train_set[:,1])), c='b', marker='o')
ax.plot_surface(x1, y1, z1, alpha=0.6)
ax = plt.title("Learning rate 1,iterasyon:100, eğitim kümesi elemanı:100")
print(len(p.s_error))
fig=plt.figure()
ax2= fig.add_subplot(111)
for i in range(len(p.s_error)):

    ax2.scatter(i+1,p.s_error[i] , c='b', marker='o')
plt.title("İterasyon başına hata miktarı.")
plt.xlabel("İterasyon")
plt.ylabel("Hata miktarı")
plt.show()
