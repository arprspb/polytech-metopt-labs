from typing import Any
from math import log, log2, ceil

class PrintableFunction():
    def __init__(self, func, readable):
        self.func = func
        self.readable = readable

    def __repr__(self):
        return self.readable
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
    

def N_steps_dichotomy_analytical(a, b, eps, delta):
    #return 2*ceil(log2((b-a)/eps))
    return 2*ceil(log2((b-a-2*delta)/(eps-delta)))

def N_steps_gr_analytical(a, b, eps):
    phi = (1+5**0.5)/2
    return ceil(log((b-a)/eps, phi)) + 2

