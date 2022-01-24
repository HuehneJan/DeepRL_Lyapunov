import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import cm
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, index):
        return self.data[index]


theta = np.arange(-np.pi, np.pi+0.01, 0.01)

thetadot = np.arange(-3, 3, 0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



xa, xb= np.meshgrid(theta, thetadot)

theta = np.reshape(xa, newshape=(-1, 1))
theta_dot = np.reshape(xb, newshape=(-1, 1))

#sin_theta = np.sin(theta)
#cos_theta = np.cos(theta)
#del theta, thetadot
#data = np.concatenate((sin_theta, cos_theta, theta_dot), axis=1).astype(np.float32)
data = np.concatenate((theta, theta_dot), axis=1).astype(np.float32)
print(data.shape)
#del sin_theta, cos_theta

torch_data = torch.from_numpy(data)
dataset = Dataset(data=torch_data)
data_loader = DataLoader(dataset=dataset, batch_size=512, shuffle=False)

epochs = np.arange(start=20000, stop=30000, step=100, dtype=int)
actions = None
for epoch in epochs:
    print(f"Epoch {epoch}")
    DIR = f"./test2/actor{epoch}.pt"
    actor = torch.load(DIR).to(device)
    current_actions = np.empty(shape=(0,1), dtype=np.float32)



    for batch in data_loader:
        batch = batch.to(device)
        batch_action = actor(batch).cpu().data.numpy()
        current_actions = np.append(current_actions, batch_action, axis=0)
    if actions is None:
        actions = current_actions
    else:
        actions = actions + current_actions
    np.save(f"./test2/array/savepont{epoch}", actions)
actions = actions / epochs.shape[0]
np.save("./test2/array/average_actions", actions)

DIR = f"./test6/actor100.pt"
actor = torch.load(DIR).to(device)
current_actions = np.empty(shape=(0, 1), dtype=np.float32)

for batch in data_loader:
    batch = batch.to(device)
    batch_action = actor(batch).cpu().data.numpy()
    current_actions = np.append(current_actions, batch_action, axis=0)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel(r'$\theta$')
current_actions = np.reshape(current_actions, newshape=xa.shape)
ax.set_ylabel(r'$\frac{d\theta}{dt}$')
ax.set_zlabel('u')
surf = ax.plot_surface(xa, xb, current_actions, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.9)
fig.colorbar(surf, ax=ax , pad=0.05, location="left", extendrect=True)
plt.show()

zero_tensor = torch.FloatTensor([[0.0, 0.0]]).to(device)
print(actor(zero_tensor))