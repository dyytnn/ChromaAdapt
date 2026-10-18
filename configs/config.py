import functools
import inspect
from typing import Callable


def _called_with_cfg(*args, **kwargs):
    """
        Check if the function is called with a CfgNode object
    """
    from omegaconf import DictConfig
    if len(args) and isinstance(args[0], DictConfig):
        return True
    if isinstance(kwargs.pop("config", None), DictConfig):
        return True
    return False

def _get_args_from_config(from_config_func: Callable, *args, **kwargs):
    """
        Use `from_config` to obtain explicit arguments
    """
    signature = inspect.signature(from_config_func)
    first_param = list(signature.parameters.keys())[0]
    if first_param not in ("config", "cfg"):
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'config' or 'cfg' as the first argument!")
    support_var_arg: bool = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        support_var_names: set[str] = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in support_var_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret
        
    
def configurable(init_func: Callable = None, *, from_config = None):
    """
        Decorate a function or class's __init__ method so that it can be called with a :class:`CfgNode` object using a :func: `from_config` function that translates :class: `CfgNode` to arguments
    """
    if init_func is not None:
        assert inspect.isfunction(init_func) and from_config is None and init_func.__name__ == "__init__", "Can only decorate __init__ method"

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = self.__class__.from_config
            except AttributeError as e:
                raise AttributeError("class with @configurable must have a from_config classmethod function") from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a from_config classmethod function")

            should_use_from_config = False

            if _called_with_cfg(*args, **kwargs):
                should_use_from_config = True
            else:
                init_signature = inspect.signature(init_func)
                init_params = list(init_signature.parameters.keys())
                if init_params and init_params[0] == 'self':
                    init_params = init_params[1:]

                provided_params = set(kwargs.keys())

                required_params = {
                    name for name, param in init_signature.parameters.items()
                    if name != 'self' and param.default == inspect.Parameter.empty
                }

                accepted_params = {
                    name for name, param in init_signature.parameters.items()
                    if name != 'self' and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                }

                if required_params and not required_params.issubset(provided_params):
                    should_use_from_config = True
                elif provided_params and not provided_params.issubset(accepted_params):
                    should_use_from_config = True

            if should_use_from_config:
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped
    else:
        if from_config is None:
            return configurable
        assert inspect.isfunction(from_config), "from_config must be a function"
        
        def wrapper(origin_func):
            @functools.wraps(origin_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return origin_func(**explicit_args)
                else:
                    return origin_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper