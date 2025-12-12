# automl_data/utils/decorators.py
"""
Кастомные декораторы.
"""

from __future__ import annotations

import functools
import time
import logging
import warnings
from typing import Any, Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')


def timing(func: Callable[P, R]) -> Callable[P, R]:
    """
    Декоратор для измерения времени выполнения.
    
    Сохраняет время в атрибуте функции last_execution_time.
    
    Example:
        >>> @timing
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()
        >>> print(slow_function.last_execution_time)  # ~1.0
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        wrapper.last_execution_time = elapsed
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} completed in {elapsed:.4f}s")
        
        return result
    
    wrapper.last_execution_time = 0.0
    return wrapper


def require_fitted(func: Callable[P, R]) -> Callable[P, R]:
    """
    Декоратор для проверки, что объект обучен.
    
    Проверяет атрибут _is_fitted перед вызовом метода.
    
    Example:
        >>> class Model:
        ...     _is_fitted = False
        ...     
        ...     @require_fitted
        ...     def predict(self, X):
        ...         return X * 2
    """
    @functools.wraps(func)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if not getattr(self, '_is_fitted', False):
            from .exceptions import NotFittedError
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )
        return func(self, *args, **kwargs)
    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Декоратор для повторных попыток при ошибках.
    
    Args:
        max_attempts: Максимальное число попыток
        delay: Начальная задержка между попытками
        backoff: Множитель увеличения задержки
        exceptions: Типы исключений для перехвата
        on_retry: Callback при повторной попытке
    
    Example:
        >>> @retry(max_attempts=3, delay=1.0)
        ... def unstable_api_call():
        ...     ...
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def deprecated(
    reason: str = "",
    version: str = "",
    replacement: str = ""
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Декоратор для пометки устаревших функций.
    
    Example:
        >>> @deprecated(reason="Use new_function", version="2.0")
        ... def old_function():
        ...     pass
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            msg = f"{func.__name__} is deprecated"
            if version:
                msg += f" since version {version}"
            if reason:
                msg += f". {reason}"
            if replacement:
                msg += f" Use {replacement} instead."
            
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_call(
    level: int = logging.DEBUG,
    log_args: bool = False,
    log_result: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Декоратор для логирования вызовов функций.
    
    Example:
        >>> @log_call(level=logging.INFO, log_args=True)
        ... def important_function(x, y):
        ...     return x + y
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger = logging.getLogger(func.__module__)
            
            msg = f"Calling {func.__name__}"
            if log_args:
                msg += f"(args={args}, kwargs={kwargs})"
            logger.log(level, msg)
            
            try:
                result = func(*args, **kwargs)
                if log_result:
                    logger.log(level, f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised: {e}")
                raise
        
        return wrapper
    return decorator


def validate_types(**type_hints: type) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Декоратор для валидации типов аргументов.
    
    Example:
        >>> @validate_types(x=int, y=str)
        ... def func(x, y):
        ...     return f"{y}: {x}"
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Получаем имена параметров
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Проверяем позиционные аргументы
            for i, arg in enumerate(args):
                if i < len(params):
                    param_name = params[i]
                    if param_name in type_hints:
                        expected = type_hints[param_name]
                        if not isinstance(arg, expected):
                            raise TypeError(
                                f"Argument '{param_name}' must be {expected.__name__}, "
                                f"got {type(arg).__name__}"
                            )
            
            # Проверяем именованные аргументы
            for name, value in kwargs.items():
                if name in type_hints:
                    expected = type_hints[name]
                    if not isinstance(value, expected):
                        raise TypeError(
                            f"Argument '{name}' must be {expected.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def singleton(cls: type) -> type:
    """
    Декоратор класса для паттерна Singleton.
    
    Example:
        >>> @singleton
        ... class Database:
        ...     pass
        >>> db1 = Database()
        >>> db2 = Database()
        >>> assert db1 is db2
    """
    instances: dict[type, Any] = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def memoize(func: Callable[P, R]) -> Callable[P, R]:
    """
    Декоратор для кэширования результатов.
    
    Example:
        >>> @memoize
        ... def fibonacci(n):
        ...     if n < 2:
        ...         return n
        ...     return fibonacci(n-1) + fibonacci(n-2)
    """
    cache: dict[tuple, Any] = {}
    
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache_clear = lambda: cache.clear()
    wrapper.cache_info = lambda: {"size": len(cache)}
    
    return wrapper


class CountCalls:
    """
    Декоратор-класс для подсчёта вызовов.
    
    Example:
        >>> @CountCalls
        ... def my_function():
        ...     pass
        >>> my_function()
        >>> my_function()
        >>> print(my_function.call_count)  # 2
    """
    
    def __init__(self, func: Callable):
        functools.update_wrapper(self, func)
        self.func = func
        self.call_count = 0
    
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)
    
    def reset(self):
        """Сброс счётчика"""
        self.call_count = 0