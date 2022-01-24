from matplotlib import pyplot as plt
import numpy as np


def calc_eigen(k1, k2):
    A = np.array([[1.0, 0.05], [0.736, 1.0]], dtype=np.float32)
    B = np.array([[0.0,0.0], [0.15*k1 , 0.15*k2]], dtype=np.float32)
    new_A = A + B
    w, v = np.linalg.eig(new_A)
    return np.abs(w)


k1 = np.load("./linear/k1_values.npy")
k2 = np.load("./linear/k2_values.npy")
k1 = k1[20000:]
k2 = k2[20000:]
print(np.mean(k1), np.mean(k2), calc_eigen(np.mean(k1), np.mean(k2)))
"""
eigen = np.empty(shape=(0,2))
for i in range(len(k1)):
    w = calc_eigen(k1[i], k2[i])
    w = np.reshape(w, newshape=(-1, 2))
    eigen = np.append(eigen, w, axis=0)
epochs = np.arange(1, 40001, 1, dtype=int)

plt.plot(epochs, eigen[:,0], label=r"$|\lambda_1|$")
plt.plot(epochs, eigen[:, 1], label="$|\lambda_2|$")
plt.grid()
plt.xlim([0,40000])
plt.ylabel(r"$|\lambda_1|, |\lambda_2|$")
plt.xlabel(r"$epoch$")
plt.legend()
plt.ylim([-0.2, 1.2])
plt.savefig("./linear/abs_eigen.svg", format="svg", transparent=False)
plt.show()



plt.plot(epochs, k1, label=r"$k1$")
plt.plot(epochs, k2, label=r"$k2$")
plt.ylabel(r"$k1, k2$")
plt.xlabel(r"$epoch$")
plt.legend()
plt.xlim([0,40000])
plt.ylim([-11, 1])
plt.grid()
plt.savefig("./linear/values.svg", format="svg", transparent=False)
plt.show()

ep_steps = np.load("./linear/epoch_steps.npy")
ep_steps = np.reshape(ep_steps, newshape=(8,5000))
ep_steps_mean = np.mean(ep_steps, axis=1)


epochs = [2500, 7500, 12500, 17500, 22500, 27500, 32500, 37500]
plt.plot(epochs, ep_steps_mean,marker="o")
plt.grid()
plt.ylabel(r"epoch length mean over 5000 epochs")
plt.xlabel(r"$epoch$")
plt.xlim([0,40000])
plt.ylim([0,210])
plt.savefig("./linear/epoch_length.svg", format="svg", transparent=False)
plt.show()
"""


