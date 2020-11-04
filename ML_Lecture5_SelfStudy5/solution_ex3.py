import torch

sigmoid = torch . nn . Sigmoid ()

x = torch . tensor (1.0 , requires_grad = True )
w1 = torch . tensor (1.0 , requires_grad = True )
w2 = torch . tensor (1.0 , requires_grad = True )
w3 = torch . tensor (1.0 , requires_grad = True )
w4 = torch . tensor (1.0 , requires_grad = True )
w5 = torch . tensor (1.0 , requires_grad = True )

y1 = sigmoid ( x * w1 )
y1 . register_hook ( lambda grad : print (" Grad y1 = { grad }"))

y2 = sigmoid ( y1 * w2 )
y2 . register_hook ( lambda grad : print (" Grad y2 = { grad }"))

y3 = sigmoid ( y2 * w3 )
y3 . register_hook ( lambda grad : print ( " Grad y3 = { grad }"))

y4 = sigmoid ( y3 * w4 )
y4 . register_hook ( lambda grad : print (  " Grad y4 = { grad }"))

y5 = sigmoid ( y4 * w5 )
y5 . register_hook ( lambda grad : print ( " Grad y5 = { grad }"))

e = (1.0 - y5 )**2

e . backward ()

print (" Grad x = {}". format ( x . grad ))
print (" Grad w1 = {}". format ( w1 . grad ))
print (" Grad w2 = {}". format ( w2 . grad ))
print (" Grad w3 = {}". format ( w3 . grad ))
print (" Grad w5 = {}". format ( w5 . grad ))
print (" Grad w4 = {}". format ( w4 . grad ))
print (" Done ")