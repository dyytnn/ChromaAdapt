from typing import Any, Dict

from tabulate import tabulate


class Registry:
    def __init__(self, name: str = "Unknown"):
        self._name = name
        self._registry: Dict[str, Any] = {}
        
    def register(self, name):
        if name in self._registry:
            print(f"{name} is already registered in {self._registry}")
            return
        
        def decorator(cls):
            self._registry[name] = cls
            return cls
        
        return decorator
    
    def build(self, cfg: dict, *args, **kwargs):
        raise NotImplementedError
    
    def get(self, name: str, *args, **kwargs):
        return self._registry.get(name)(*args, **kwargs)
    
    def check(self, name: str):
        return name in self._registry
    
    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._registry.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table
    
class NormalRegistry:
    def __init__(self, register: Registry, module):
        super().__init__()
        self.register = register
        self.module = module
    
    def __call__(self, register_name):
        @self.register.register(register_name)
        class _Register(self.module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __repr__(self):
                return f"{register_name}()"
        return _Register
        