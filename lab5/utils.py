


class Func():
    def __init__(f, grad):
        self.f = f
        self.grad = grad

    def __call__(self,*args):
        return self.f(*args)
    
    def grad():
        return self.grad(*args)
        
f = Func(lambda *args: args[0]**2+args[1]**2, lambda *args: (2*args[0], 2*args[1]))

print(f(5,2))
print(f.grad(5,2))