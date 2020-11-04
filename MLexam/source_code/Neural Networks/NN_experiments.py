import torch
import torch.nn as nn
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
import time

'''This code was purely for more in depth experiments and to understand the underlying concepts of backpropagation, based on the code from selfstudy 5'''

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    #pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    print(x_train.shape)


x_train, y_train, x_valid, y_valid, x_test, y_test = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

weights = torch.randn(784, 10) / np.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def softmax(x):
    return x.exp() / x.exp().sum(-1).unsqueeze(-1)

# Below @ refers to matrix multiplication
def model(xb):
    return softmax(xb @ weights + bias)

batch_size = 64

#def nll(input, target):
    #return (-input[range(target.shape[0]), target].log()).mean()

loss_func = nn.CrossEntropyLoss()

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

epochs = 30  # how many epochs to train for
lr = 0.01 # learning rate

start = time.time()
train_losses = []
valid_losses = []
momentum = 0.9
v = 0
v_bias = 0


for epoch in range(epochs):
    for batch_idx in range((n - 1) // batch_size + 1):
        start_i = batch_idx * batch_size
        end_i = start_i + batch_size
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            #v = momentum * v + lr *(weights.grad) ##momentum
            #weights-=v
            #v_bias = momentum * v_bias -lr *(bias.grad)
            bias -=  bias.grad*lr
            weights.grad.zero_()
            bias.grad.zero_()
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    train_loss = loss_func(model(x_train), y_train)
                    print(f"Epoch: {epoch}, B-idx: {batch_idx}, Training loss: {train_loss}")
                    train_losses.append(train_loss)
                    valid_loss = loss_func(model(x_valid), y_valid)
                    print(f"Epoch: {epoch}, B-idx: {batch_idx}, valid loss: {valid_loss}")
                    valid_losses.append(train_loss)


plt.plot(range(len(train_losses)), train_losses,'b')
plt.show()

pred_labels=model(x_test)

print(f"Accuracy of model on batch (with random weights): {accuracy(pred_labels, y_test)}")
