import inspect
import logging
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from rich.console import Console
from rich.table import Table

from ...configs import Config, ConfigDict


def get_object_from_string(obj_name: str):
    """Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    parts = iter(obj_name.split('.'))
    module_name = next(parts)
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            obj = getattr(module, part, None)
            if obj is not None and not inspect.ismodule(obj):
                break
            module_name = f'{module_name}.{part}'
        except StopIteration:
            return module
        except ImportError:
            return None
        
    obj = module
    while True:
        try:
            obj = getattr(obj, part)
            part = next(parts)
        except StopIteration:
            return obj
        except AttributeError:
            return None
        
def is_seq_of(
    seq: Any,
    expected_type: Union[Type, Tuple],
    seq_type: Optional[Type] = None,
) -> bool:
    """Check whether the seq is a sequence of some type
    
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def build_from_cfg(cfg: Union[dict, ConfigDict], registry: Registry, default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(f'cfg must be a dict or a ConfigDict, but got {type(cfg)}')
    
    if "type" not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
            f'but got {type(default_args)}')
        
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
            
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        pass
    pass

class Registry:
    def __init__(
        self, 
        name: str,
        build_func: Optional[Callable] = None,
        parent: Optional['Registry'] = None,
        scope: Optional[str] = None,
        locations: List[str] = []
    ) -> None:
        self._name = name
        self._module_dict: Dict[str, Type] = dict()
        self._children: Dict[str, 'Registry'] = dict()
        self._location = locations
        self._imported = False
        
        if scope is not None:
            assert isinstance(scope, str)
            self._scope = scope
        else:
            self._scrope = self.infer_scope()

        self.parent: Optional['Registry']
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_child(self)
            self.parent = parent
        else:
            self.parent = None
        
        self.build_func: Callable
        if build_func is None:
            if self.parent is not None:
                self.build_func = self.parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
    
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    def __repr__(self):
        table = Table(title=f"Registry of {self._name}")
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Objects", justify="left", style="green")
        
        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))
        
        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()
    
    
    @staticmethod
    def _add_child(child: 'Registry'):
        pass
    
    @staticmethod
    def infer_scope(self) -> str:
        module = inspect.getmodule(sys._getframe(2))
        if module is not None:
            filename = module.__name__
            split_filename = filename.split('.')
            scope = split_filename[0]
        else:
            scope = f"Cannot infer the scope of {module.__name__}"
            print(scope)
        return scope
    
    @staticmethod
    def split_scope_key(key: str) -> Tuple[Optional[str], str]:
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index], key[:split_index + 1:] 
        else:
            return None, key
        
    @property
    def name(self):
        return self._name
    
    @property
    def scope(self):
        return self._scope
    
    @property
    def module_dict(self):
        return self._module_dict
    
    @property
    def children(self):
        return self._children
    
    @property
    def root(self):
        return self._get_root_registry()
    
    @contextmanager
    def switch_scope_and_registry(self, scope: Optional[str]) -> Generator:
        raise NotImplementedError
    
    def _get_root_registry(self) -> 'Registry':
        root = self
        while root.parent is not None:
            root = root.parent
        
        return root
    
    def import_from_location(self) -> None:
        pass
    
        