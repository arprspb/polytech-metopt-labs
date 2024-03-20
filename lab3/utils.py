from typing import Any

class PrintableFunction():
    def __init__(self, func, readable):
        self.func = func
        self.readable = readable

    def __repr__(self):
        return self.readable
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)