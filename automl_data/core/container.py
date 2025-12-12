# automl_data/core/container.py
"""
DataContainer — универсальный контейнер для любых типов данных.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Iterator, Self, Callable, Sequence
from pathlib import Path
import uuid
import copy

import pandas as pd
import numpy as np


class DataType(Enum):
    """Типы данных"""
    TABULAR = auto()      # Табличные данные
    TEXT = auto()         # Текст (NLP)
    IMAGE = auto()        # Изображения (CV)
    MULTIMODAL = auto()   # Смешанные данные


class ProcessingStage(Enum):
    """Стадии обработки"""
    RAW = auto()
    PROFILED = auto()
    VALIDATED = auto()
    CLEANED = auto()
    AUGMENTED = auto()
    TRANSFORMED = auto()
    READY = auto()
    
    def __lt__(self, other: ProcessingStage) -> bool:
        return self.value < other.value


@dataclass
class ProcessingStep:
    """Шаг обработки"""
    name: str
    component: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    input_shape: tuple | None = None
    output_shape: tuple | None = None
    duration_seconds: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.name} ({self.duration_seconds:.2f}s)"


@dataclass
class DataContainer:
    """
    Универсальный контейнер данных.
    
    Поддерживает:
    - Табличные данные (DataFrame)
    - Текстовые данные (список строк или DataFrame с текстовой колонкой)
    - Изображения (пути к файлам или numpy arrays)
    
    Examples:
        # Табличные данные
        >>> container = DataContainer(df, target_column="label")
        
        # Текстовые данные
        >>> container = DataContainer(
        ...     df, 
        ...     text_column="review",
        ...     target_column="sentiment",
        ...     data_type=DataType.TEXT
        ... )
        
        # Изображения
        >>> container = DataContainer(
        ...     df,
        ...     image_column="path",
        ...     target_column="class",
        ...     data_type=DataType.IMAGE
        ... )
    """
    
    data: pd.DataFrame
    target_column: str | None = None
    
    # Для текстовых данных
    text_column: str | None = None
    
    # Для изображений
    image_column: str | None = None
    image_dir: Path | None = None
    
    # Метаданные
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    data_type: DataType = DataType.TABULAR
    stage: ProcessingStage = ProcessingStage.RAW
    created_at: datetime = field(default_factory=datetime.now)
    source: str | None = None
    
    # История
    processing_history: list[ProcessingStep] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    
    # Кэш для аугментированных данных
    _augmented_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Валидация и автоопределение типа"""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"data must be DataFrame, got {type(self.data).__name__}")
        
        # Автоопределение типа данных
        if self.data_type == DataType.TABULAR:
            self.data_type = self._infer_data_type()
        
        # Конвертация путей
        if self.image_dir and isinstance(self.image_dir, str):
            self.image_dir = Path(self.image_dir)
    
    # ==================== ПЕРЕГРУЗКА ОПЕРАТОРОВ ====================
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __bool__(self) -> bool:
        return len(self.data) > 0
    
    def __contains__(self, column: str) -> bool:
        return column in self.data.columns
    
    def __getitem__(self, key: str | int | slice | list) -> Self | pd.Series:
        match key:
            case str():
                return self.data[key]
            case int():
                return self._subset(self.data.iloc[[key]])
            case slice():
                return self._subset(self.data.iloc[key])
            case list() if all(isinstance(k, str) for k in key):
                return self._subset(self.data[key])
            case _:
                raise TypeError(f"Invalid key type: {type(key)}")
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
    
    def __iter__(self) -> Iterator[pd.Series]:
        for _, row in self.data.iterrows():
            yield row
    
    def __add__(self, other: Self) -> Self:
        """Объединение контейнеров"""
        if not isinstance(other, DataContainer):
            return NotImplemented
        combined = pd.concat([self.data, other.data], ignore_index=True)
        return self._subset(combined)
    
    def __repr__(self) -> str:
        return (
            f"DataContainer(type={self.data_type.name}, shape={self.shape}, "
            f"stage={self.stage.name}, quality={self.quality_score:.0%})"
        )
    
    # ==================== PROPERTIES ====================
    
    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape
    
    @property
    def columns(self) -> list[str]:
        return self.data.columns.tolist()
    
    @property
    def numeric_columns(self) -> list[str]:
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    @property
    def categorical_columns(self) -> list[str]:
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    @property
    def X(self) -> pd.DataFrame:
        """Признаки (без target)"""
        cols_to_drop = [c for c in [self.target_column] if c and c in self.data.columns]
        return self.data.drop(columns=cols_to_drop)
    
    @property
    def y(self) -> pd.Series | None:
        """Целевая переменная"""
        if self.target_column and self.target_column in self.data.columns:
            return self.data[self.target_column]
        return None
    
    @property
    def texts(self) -> pd.Series | None:
        """Текстовые данные"""
        if self.text_column and self.text_column in self.data.columns:
            return self.data[self.text_column]
        return None
    
    @property
    def image_paths(self) -> list[Path] | None:
        """Пути к изображениям"""
        if self.image_column and self.image_column in self.data.columns:
            paths = self.data[self.image_column].tolist()
            if self.image_dir:
                return [self.image_dir / p for p in paths]
            return [Path(p) for p in paths]
        return None
    
    @property
    def is_text(self) -> bool:
        return self.data_type == DataType.TEXT
    
    @property
    def is_image(self) -> bool:
        return self.data_type == DataType.IMAGE
    
    @property
    def is_tabular(self) -> bool:
        return self.data_type == DataType.TABULAR
    
    @property
    def class_distribution(self) -> dict[Any, int] | None:
        """Распределение классов"""
        if self.y is not None:
            return self.y.value_counts().to_dict()
        return None
    
    imbalance_threshold: float = 0.3
    
    @property
    def is_imbalanced(self) -> bool:
        """Проверка на дисбаланс классов с учетом порога"""
        if self.class_distribution:
            counts = list(self.class_distribution.values())
            if len(counts) >= 2:
                ratio = min(counts) / max(counts)
                return ratio < self.imbalance_threshold
        return False
    
    # ==================== МЕТОДЫ ====================
    
    def clone(self) -> Self:
        """Глубокая копия"""
        return DataContainer(
            data=self.data.copy(),
            target_column=self.target_column,
            text_column=self.text_column,
            image_column=self.image_column,
            image_dir=self.image_dir,
            id=f"{self.id}_clone",
            data_type=self.data_type,
            stage=self.stage,
            source=self.source,
            processing_history=copy.deepcopy(self.processing_history),
            profile=copy.deepcopy(self.profile),
            quality_score=self.quality_score,
            recommendations=copy.deepcopy(self.recommendations)
        )
    
    def add_step(self, step: ProcessingStep) -> None:
        """Добавить шаг в историю"""
        self.processing_history.append(step)
    
    def split(
        self, 
        train_ratio: float = 0.8,
        random_state: int = 42,
        stratify: bool = True
    ) -> tuple[Self, Self]:
        """Разделение на train/test с стратификацией"""
        from sklearn.model_selection import train_test_split
        
        strat = None
        if stratify and self.target_column and self.target_column in self.data.columns:
            strat = self.data[self.target_column]
        
        train_df, test_df = train_test_split(
            self.data,
            train_size=train_ratio,
            random_state=random_state,
            stratify=strat
        )
        
        return self._subset(train_df), self._subset(test_df)
    
    def get_splits(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Получить X_train, X_test, y_train, y_test"""
        train, test = self.split(1 - test_size, random_state)
        return train.X, test.X, train.y, test.y
    
    def sample(self, n: int | None = None, frac: float | None = None) -> Self:
        """Случайная выборка"""
        sampled = self.data.sample(n=n, frac=frac)
        return self._subset(sampled)
    
    def filter(self, condition: pd.Series | Callable) -> Self:
        """Фильтрация"""
        if callable(condition):
            mask = condition(self.data)
        else:
            mask = condition
        return self._subset(self.data[mask])
    
    def summary(self) -> dict[str, Any]:
        """Сводка"""
        info = {
            "id": self.id,
            "type": self.data_type.name,
            "shape": self.shape,
            "stage": self.stage.name,
            "quality": f"{self.quality_score:.0%}",
            "target": self.target_column,
        }
        
        if self.is_text:
            info["text_column"] = self.text_column
            if self.texts is not None:
                info["avg_text_length"] = int(self.texts.str.len().mean())
        
        if self.is_image:
            info["image_column"] = self.image_column
            info["image_dir"] = str(self.image_dir) if self.image_dir else None
        
        if self.class_distribution:
            info["classes"] = len(self.class_distribution)
            info["imbalanced"] = self.is_imbalanced
        
        return info
    
    def _subset(self, data: pd.DataFrame) -> Self:
        """Создание подмножества"""
        return DataContainer(
            data=data.reset_index(drop=True),
            target_column=self.target_column,
            text_column=self.text_column,
            image_column=self.image_column,
            image_dir=self.image_dir,
            data_type=self.data_type,
            stage=self.stage,
            source=self.source
        )
    
    def _infer_data_type(self) -> DataType:
        """Автоопределение типа данных"""
        # Явно указан текст
        if self.text_column:
            return DataType.TEXT
        
        # Явно указаны изображения
        if self.image_column:
            return DataType.IMAGE
        
        # Проверяем на длинные тексты
        for col in self.categorical_columns[:5]:
            try:
                avg_len = self.data[col].dropna().astype(str).str.len().mean()
                if avg_len > 100:
                    self.text_column = col
                    return DataType.TEXT
            except:
                pass
        
        # Проверяем на пути к файлам
        for col in self.categorical_columns[:5]:
            try:
                sample = self.data[col].dropna().iloc[0]
                if isinstance(sample, str) and any(
                    sample.lower().endswith(ext) 
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                ):
                    self.image_column = col
                    return DataType.IMAGE
            except:
                pass
        
        return DataType.TABULAR
    
    def _sync_internal_state(self):
        """
        Синхронизирует все кэшированные свойства после изменения данных.
        КРИТИЧЕСКИ ВАЖНО вызывать после ЛЮБОГО изменения container.data!
        """
        # Удаляем все кэши
        cache_attrs = [
            '_X_cache', '_y_cache', '_numeric_columns_cache',
            '_categorical_columns_cache', '_text_columns_cache',
            '_image_columns_cache', '_quality_score_cache',
            '_processing_history_cache', '_profile_cache'
        ]
        
        for attr in cache_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Форсируем пересчет при следующем обращении
        self._force_recompute = True
        
        # Также сбрасываем любые другие вычисленные свойства
        self._computed_properties = {}