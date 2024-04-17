class Func():
    def __init__(self,f, grad):
        self.f = f
        self.grad = grad

    def __call__(self,*args):
        return self.f(*args)
    
    def grad(self, *args):
        return self.grad(*args)
        
f = Func(lambda x: x[0]**2+x[1]**2, lambda x: (2*x[0], 2*x[1]))

print(f([5,2]))
print(f.grad([5,2]))