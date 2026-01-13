# automl_data/utils/decorators.py
"""
Кастомные декораторы для библиотеки.
"""

from __future__ import annotations

import functools
import time
import logging
from typing import Any, Callable, TypeVar, ParamSpec, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.container import DataContainer

P = ParamSpec('P')
R = TypeVar('R')


def timing(func: Callable[P, R]) -> Callable[P, R]:
    """
    Декоратор для измерения времени выполнения функции.
    
    Сохраняет время в атрибуте функции last_execution_time.
    
    Example:
        >>> @timing
        ... def slow_function():
        ...     time.sleep(0.1)
        >>> slow_function()
        >>> print(slow_function.last_execution_time)  # ~0.1
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


def timing_method(attr_name: str = '_last_duration') -> Callable:
    """
    Декоратор для измерения времени выполнения метода класса.
    
    Сохраняет время в атрибут экземпляра.
    
    Args:
        attr_name: Имя атрибута для сохранения времени
    
    Example:
        >>> class MyClass:
        ...     _last_duration = 0.0
        ...     
        ...     @timing_method('_last_duration')
        ...     def process(self):
        ...         time.sleep(0.1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            
            setattr(self, attr_name, elapsed)
            
            logger = logging.getLogger(func.__module__)
            logger.debug(f"{self.__class__.__name__}.{func.__name__} completed in {elapsed:.4f}s")
            
            return result
        return wrapper
    return decorator



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
    def wrapper(self, *args, **kwargs):
        if not getattr(self, '_is_fitted', False):
            from .exceptions import NotFittedError
            
            name = getattr(self, '_name', None) or self.__class__.__name__
            
            raise NotFittedError(
                f"{name} is not fitted. Call fit() first.",
                component=name
            )
        return func(self, *args, **kwargs)
    return wrapper



def safe_transform(
    preserve_target: bool = True,
    sync_state: bool = True,
    reset_index: bool = False
) -> Callable:
    """
    Комбинированный декоратор для безопасной трансформации данных.
    
    Args:
        preserve_target: Сохранять target колонку без изменений
        sync_state: Вызывать _sync_internal_state после трансформации
        reset_index: Сбрасывать индекс DataFrame
    
    Example:
        >>> class MyAdapter(BaseAdapter):
        ...     @safe_transform(preserve_target=True)
        ...     def _transform_impl(self, container):
        ...         container.data = container.data.dropna()
        ...         return container
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, container):
            import numpy as np
            
            target_col = getattr(container, 'target_column', None)
            original_target = None
            original_len = len(container.data) if hasattr(container, 'data') else 0
            
            # 1. Сохраняем target
            if preserve_target and target_col and target_col in container.data.columns:
                original_target = container.data[target_col].copy()
            
            # 2. Выполняем трансформацию
            result = func(self, container)
            
            # 3. Reset index
            if reset_index and hasattr(result, 'data'):
                result.data = result.data.reset_index(drop=True)
            
            # 4. Восстанавливаем target (только если длина совпадает)
            if (original_target is not None and 
                hasattr(result, 'data') and
                target_col in result.data.columns and
                len(result.data) == original_len):
                try:
                    current = result.data[target_col].values
                    original = original_target.values
                    if not np.array_equal(current, original):
                        result.data[target_col] = original
                except (ValueError, TypeError):
                    pass  # Игнорируем ошибки сравнения
            
            # 5. Синхронизируем
            if sync_state and hasattr(result, '_sync_internal_state'):
                result._sync_internal_state()
            
            return result
        return wrapper
    return decorator


def preserve_target(func: Callable) -> Callable:
    """
    Декоратор для сохранения target колонки без изменений.
    
    Example:
        >>> @preserve_target
        ... def _transform_impl(self, container):
        ...     # Модифицируем данные
        ...     return container
    """
    @functools.wraps(func)
    def wrapper(self, container):
        import numpy as np
        
        target_col = getattr(container, 'target_column', None)
        original_target = None
        original_len = len(container.data) if hasattr(container, 'data') else 0
        
        if target_col and hasattr(container, 'data') and target_col in container.data.columns:
            original_target = container.data[target_col].copy()
        
        result = func(self, container)
        
        if (original_target is not None and 
            hasattr(result, 'data') and
            target_col in result.data.columns and
            len(result.data) == original_len):
            try:
                if not np.array_equal(result.data[target_col].values, original_target.values):
                    result.data[target_col] = original_target.values
            except (ValueError, TypeError):
                pass
        
        return result
    return wrapper


def sync_container(func: Callable) -> Callable:
    """
    Декоратор для автоматической синхронизации контейнера после трансформации.
    
    Example:
        >>> @sync_container
        ... def _transform_impl(self, container):
        ...     container.data = modified_df
        ...     return container
    """
    @functools.wraps(func)
    def wrapper(self, container):
        result = func(self, container)
        
        if hasattr(result, '_sync_internal_state'):
            result._sync_internal_state()
        
        return result
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
        delay: Начальная задержка между попытками (секунды)
        backoff: Множитель увеличения задержки
        exceptions: Типы исключений для перехвата
        on_retry: Callback при повторной попытке (exception, attempt_number)
    
    Example:
        >>> @retry(max_attempts=3, delay=0.1)
        ... def unstable_function():
        ...     # Может упасть
        ...     pass
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


# ==================== SINGLETON DECORATOR ====================

def singleton(cls: type) -> type:
    """
    Декоратор класса для паттерна Singleton.
    
    Example:
        >>> @singleton
        ... class Database:
        ...     def __init__(self):
        ...         self.connection = "connected"
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


class CountCalls:
    """
    Декоратор-класс для подсчёта вызовов функции.
    
    Example:
        >>> @CountCalls
        ... def my_function():
        ...     return "called"
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
    
    def reset(self) -> None:
        """Сброс счётчика вызовов."""
        self.call_count = 0