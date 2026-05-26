# import numpy as np

# x = int(input("Enter the no. of Bathrooms (/Rs. 5k):"))
# y = int(input("Enter the no. of kitchens (/Rs. 6k):"))
# z = int(input("Enter the no. of Bedrooms (/Rs. 15k):"))

# # Price of per bathroom, kitchen, bedroom in Rs. k
# w = np.array([5, 6, 15])

# # No. of bathrooms, kitchens, bedrooms
# v = np.array([x,y,z])

# p = np.dot(w,v)

# print(f"Total cost for the house: Rs.{p}k")

import torch
import torch.nn as nn

x = torch.tensor([[15., 3., 5.]])   # shape (1, 3)
y = torch.tensor([[690.]])           # true price

layer = nn.Linear(3, 1, bias=False)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

for epoch in range(100):
    # 1. Forward pass
    y_hat = layer(x)

    # 2. Compute loss
    loss = loss_fn(y_hat, y)

    # 3. Backward pass — PyTorch computes all ∂L/∂W automatically
    optimizer.zero_grad()   # clear gradients from last step
    loss.backward()         # fill .grad on every parameter

    # 4. Update W
    optimizer.step()        # W = W - lr * W.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.1f}, W={layer.weight.data}")