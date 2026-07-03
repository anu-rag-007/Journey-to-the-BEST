###### **## MNIST Digit Classifier**

###### 

###### **Built a neural network from scratch in two stages:**

###### 

###### **### Stage 1 — NumPy (no frameworks)**

###### **- 3-layer network: 784 → 128 → 64 → 10**

###### **- Manual forward pass, backpropagation, gradient descent**

###### **- Test accuracy: \*\*97.88%\*\***

###### 

###### **### Stage 2 — PyTorch**

###### **- 4-layer network: 784 → 512 → 256 → 128 → 10**

###### **- Adam optimizer, dropout, LR scheduling**

###### **- Test accuracy: \*\*98.47%\*\***

###### 

###### **### Key insight**

###### **Wider network (5x more parameters) improved accuracy by 0.59%.**

###### **BatchNorm added unnecessary complexity for this dataset.**

###### **LR scheduler helped fine-tune in later epochs.**

