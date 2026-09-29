import torch
def f(x):
    return 3*x**2-4*x
h=0.0001
x0=1.0
numerical=(f(x0+h)-f(x0))/h
print(numerical)
print(6*x0-4)
x=torch.arange(4.0,requires_grad=True)
print(x)
print(x.grad)
y=2*torch.dot(x,x)
print(y)
y.backward()
print(x.grad)
y=x.sum()
y.backward()
print(x.grad)
x.grad.zero_()
y=x.sum()
y.backward()
print(x.grad)
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
z.sum().backward()
print(x.grad)
def f2(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c
a=torch.randn(size=(),requires_grad=True)
d=f2(a)
d.backward()
print(a.grad)
print(a.grad==d/a)
