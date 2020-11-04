import torch
from math import log

nb_classes = 2

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()
logloss = torch.nn.BCELoss()

x1 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(2.0, requires_grad=True)

x2 = torch.tensor(3.0, requires_grad=True)
w2 = torch.tensor(0.5, requires_grad=True)

w0 = torch.tensor(1.5, requires_grad=True)


y1 = x1*w1
print(y1)
y1.register_hook(lambda grad: print("Grad y1 = {}".format(grad)))


y2 = x2*w2
print(y2)
y2.register_hook(lambda grad: print("Grad y2 = {}".format(grad)))

y3 = y1+y2
print(y3)
y3.register_hook(lambda grad: print("Grad y3 = {}".format(grad)))

h = relu(y3)
print(h)
h.register_hook(lambda grad: print("Grad h = {}".format(grad)))

#logistic function missing as an activation one

y4 = h * w0
print(y4)
y4.register_hook(lambda grad: print("Grad y4 = {}".format(grad)))

y5 = sigmoid(y4)
print(y5)
y5.register_hook(lambda grad: print("Grad y5 = {}".format(grad)))

target = torch.tensor(1, dtype=torch.float)

e = logloss(y5,target)
print(e)


e.backward()

##log_loss should be with log_loss(predicted,actual)



print("Grad x1 = {}".format(x1.grad))
print("Grad x2 = {}".format(x2.grad))
print("Grad w1 = {}".format(w1.grad))
print("Grad w2 = {}".format(w2.grad))
print("Grad w0 = {}".format(w0.grad))


print("Done")