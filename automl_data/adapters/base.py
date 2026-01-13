# automl_data/adapters/base.py
"""
Базовые классы адаптеров.

Демонстрирует:
- ABC и абстрактные методы
- Паттерн Template Method
- Наследование
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from dataclasses import dataclass
import logging
import time

from ..core.container import DataContainer, ProcessingStep, ProcessingStage
from ..utils.decorators import timing_method, require_fitted

T = TypeVar('T')

class BaseAdapter(ABC):
    """
    Абстрактный базовый класс для всех адаптеров.
        
    Все адаптеры должны реализовать _fit_impl и _transform_impl.
    
    Example:
        >>> class MyAdapter(BaseAdapter):
        ...     def _fit_impl(self, container):
        ...         # Логика обучения
        ...         pass
        ...     
        ...     def _transform_impl(self, container):
        ...         # Логика трансформации
        ...         return container
    """
    
    def __init__(self, name: str | None = None, **config):
        self._name = name or self.__class__.__name__
        self._logger = logging.getLogger(f"automl_data.{name}")
        self._config = config
        self._is_fitted = False
        self._fit_info: dict[str, Any] = {}
        self._last_duration: float = 0.0
    
    @property
    def name(self) -> str:
        """Имя адаптера"""
        return self._name
    
    @property
    def is_fitted(self) -> bool:
        """Обучен ли адаптер"""
        return self._is_fitted
    
    @property
    def fit_info(self) -> dict[str, Any]:
        """Информация об обучении"""
        return self._fit_info.copy()
    
    @abstractmethod
    def _fit_impl(self, container: DataContainer) -> None:
        """
        Реализация обучения.
        Переопределяется в подклассах.
        """
        pass
    
    @abstractmethod
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        """
        Реализация трансформации.
        Переопределяется в подклассах.
        """
        pass
    
    @timing_method('_last_duration')
    def fit(self, container: DataContainer) -> BaseAdapter:
        """
        Template Method для обучения.
        
        1. Валидация входных данных
        2. Pre-fit hook
        3. Вызов _fit_impl
        4. Post-fit hook
        5. Установка флага _is_fitted
        
        Args:
            container: DataContainer с данными
        
        Returns:ф
            self (для цепочки вызовов)
        """
        
        self._validate_input(container)
        self._pre_fit(container)
        self._fit_impl(container)
        self._post_fit(container)
        self._is_fitted = True
         
        return self
    
    @require_fitted
    def transform(self, container: DataContainer) -> DataContainer:
        """
        Template Method для трансформации.
        
        1. Проверка is_fitted
        2. Валидация входных данных
        3. Вызов _transform_impl
        4. Запись в историю
        
        Args:
            container: DataContainer с данными
        
        Returns:
            Трансформированный DataContainer
        """
        start = time.perf_counter()
        
        self._validate_input(container)
        result = self._transform_impl(container)
        
        duration = time.perf_counter() - start
        self._add_to_history(result, duration)
        
        return result
    
    def fit_transform(self, container: DataContainer) -> DataContainer:
        """
        Fit и Transform в одном вызове.
        
        Args:
            container: DataContainer с данными
        
        Returns:
            Трансформированный DataContainer
        """
        self.fit(container)
        return self.transform(container)
    
    def _validate_input(self, container: DataContainer) -> None:
        """
        Валидация входных данных.
        Может быть переопределён в подклассах.
        """
        if container is None:
            from ..utils.exceptions import ValidationError
            raise ValidationError("Container cannot be None")
        
        if len(container) == 0:
            from ..utils.exceptions import ValidationError
            raise ValidationError("Container data is empty")
    
    def _pre_fit(self, container: DataContainer) -> None:
        """
        Hook перед обучением.
        Может быть переопределён в подклассах.
        """
        pass
    
    def _post_fit(self, container: DataContainer) -> None:
        """
        Hook после обучения.
        Может быть переопределён в подклассах.
        """
        pass
    
    def _add_to_history(self, container: DataContainer, duration: float) -> None:
        """Добавление шага в историю контейнера"""
        step = ProcessingStep(
            name=self._name,
            component=self.__class__.__name__,
            params=self.get_params(),
            input_shape=container.shape,
            output_shape=container.shape,
            duration_seconds=duration
        )
        container.add_step(step)
    
    def get_params(self) -> dict[str, Any]:
        """Получить параметры адаптера"""
        return {
            "name": self._name,
            "class": self.__class__.__name__,
            **self._config,
            **self._fit_info
        }
    
    def set_params(self, **params) -> BaseAdapter:
        """Установить параметры"""
        for key, value in params.items():
            if key in self._config:
                self._config[key] = value
            elif hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def clone(self) -> BaseAdapter:
        """Создать копию адаптера (без состояния fit)"""
        return self.__class__(name=self._name, **self._config)
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self._name}', {status})"
    
    def __str__(self) -> str:
        return f"{self._name} ({self.__class__.__name__})"


class TransformOnlyAdapter(BaseAdapter):
    """
    Адаптер, не требующий обучения.
    
    Используется для операций, которые не имеют состояния:
    - Простые преобразования
    - Фильтрация
    - Анализ
    """
    
    def _fit_impl(self, container: DataContainer) -> None:
        """Обучение не требуется"""
        pass
    
    def transform(self, container: DataContainer) -> DataContainer:
        """
        Transform без проверки is_fitted.
        """
        start = time.perf_counter()
        
        self._validate_input(container)
        result = self._transform_impl(container)
        
        duration = time.perf_counter() - start
        self._add_to_history(result, duration)
        
        return result
    
    def fit_transform(self, container: DataContainer) -> DataContainer:
        """Просто transform, fit не нужен"""
        self._is_fitted = True
        return self.transform(container)
