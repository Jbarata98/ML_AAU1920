import torch

sigmoid = torch.nn.Sigmoid()


x1 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
w3 = torch.tensor(1.0, requires_grad=True)
w4 = torch.tensor(0.5, requires_grad=True)
w5 = torch.tensor(0.5, requires_grad=True)


y1 = x1*w1
y1.register_hook(lambda grad: print("Grad y1 = {}".format(grad)))

y2 = sigmoid(y1)
y2.register_hook(lambda grad: print("Grad y2 = {}".format(grad)))

y3 = y2 * w2
y3.register_hook(lambda grad: print("Grad y3 = {}".format(grad)))

y4 = sigmoid(y3)
y4.register_hook(lambda grad: print("Grad y4 = {}".format(grad)))

y5 = y4 * w3
y5.register_hook(lambda grad: print("Grad y5 = {}".format(grad)))

y6 = sigmoid(y5)
y6.register_hook(lambda grad: print("Grad y6 = {}".format(grad)))

y7 = y6 * w4
y7.register_hook(lambda grad: print("Grad y7 = {}".format(grad)))

y8 = sigmoid(y7)
y8.register_hook(lambda grad: print("Grad y8 = {}".format(grad)))

y9 = y8 * w5
y9.register_hook(lambda grad: print("Grad y9 = {}".format(grad)))

####
y11 = sigmoid(y9)
y11.register_hook(lambda grad: print("Grad y11 = {}".format(grad)))


e = (1-y11)**2

e.backward()

print("Grad x1 = {}".format(x1.grad))
print("Grad w1 = {}".format(w1.grad))
print("Grad w2 = {}".format(w2.grad))
print("Grad w3 = {}".format(w3.grad))
print("Grad w4 = {}".format(w4.grad))
print("Grad w5 = {}".format(w5.grad))



print("Done")