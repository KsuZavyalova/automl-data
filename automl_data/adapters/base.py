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
from ..utils.decorators import timing, require_fitted

T = TypeVar('T')


@dataclass
class AdapterResult:
    """Результат работы адаптера"""
    success: bool
    message: str = ""
    details: dict[str, Any] = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseAdapter(ABC):
    """
    Абстрактный базовый класс для всех адаптеров.
    
    Использует паттерн Template Method:
    - fit() вызывает _fit_impl()
    - transform() вызывает _transform_impl()
    
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
    
    # ==================== ABSTRACT METHODS ====================
    
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
    
    # ==================== TEMPLATE METHODS ====================
    
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
        
        Returns:
            self (для цепочки вызовов)
        """
        start = time.perf_counter()
        
        self._validate_input(container)
        self._pre_fit(container)
        self._fit_impl(container)
        self._post_fit(container)
        self._is_fitted = True
        
        self._last_duration = time.perf_counter() - start
        
        return self
    
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
        if not self._is_fitted:
            from ..utils.exceptions import NotFittedError
            raise NotFittedError(
                f"{self._name} is not fitted. Call fit() first.",
                component=self._name
            )
        
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
    
    # ==================== HOOKS ====================
    
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
    
    # ==================== UTILITY METHODS ====================
    
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


class CompositeAdapter(BaseAdapter):
    """
    Композитный адаптер (паттерн Composite).
    Объединяет несколько адаптеров в последовательность.
    
    Example:
        >>> composite = CompositeAdapter()
        >>> composite.add(EncodingAdapter())
        >>> result = composite.fit_transform(container)
    """
    
    def __init__(self, adapters: list[BaseAdapter] | None = None, **config):
        super().__init__(name="CompositeAdapter", **config)
        self._adapters: list[BaseAdapter] = adapters or []
    
    def add(self, adapter: BaseAdapter) -> CompositeAdapter:
        """
        Добавить адаптер (fluent API).
        
        Args:
            adapter: Адаптер для добавления
        
        Returns:
            self (для цепочки вызовов)
        """
        self._adapters.append(adapter)
        return self
    
    def _fit_impl(self, container: DataContainer) -> None:
        """Обучение всех вложенных адаптеров"""
        current = container
        for adapter in self._adapters:
            adapter.fit(current)
            # Трансформируем для следующего адаптера
            current = adapter.transform(current)
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        """Последовательное применение всех адаптеров"""
        current = container
        for adapter in self._adapters:
            current = adapter.transform(current)
        return current
    
    def __len__(self) -> int:
        return len(self._adapters)
    
    def __iter__(self):
        return iter(self._adapters)
    
    def __getitem__(self, index: int) -> BaseAdapter:
        return self._adapters[index]


class ConditionalAdapter(BaseAdapter):
    """
    Адаптер с условным выполнением.
    
    Выполняет внутренний адаптер только если условие True.
    
    Example:
        >>> adapter = ConditionalAdapter(
        ...     adapter=BalancingAdapter(),
        ...     condition=lambda c: c.is_imbalanced
        ... )
    """
    
    def __init__(
        self, 
        adapter: BaseAdapter,
        condition: callable,
        **config
    ):
        super().__init__(name=f"Conditional({adapter.name})", **config)
        self._adapter = adapter
        self._condition = condition
        self._condition_met = False
    
    def _fit_impl(self, container: DataContainer) -> None:
        self._condition_met = self._condition(container)
        if self._condition_met:
            self._adapter.fit(container)
    
    def _transform_impl(self, container: DataContainer) -> DataContainer:
        if self._condition_met:
            return self._adapter.transform(container)
        return container
    
    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params["condition_met"] = self._condition_met
        params["inner_adapter"] = self._adapter.get_params()
        return params