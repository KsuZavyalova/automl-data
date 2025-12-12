# automl_data/__init__.py
"""
ML Data Forge — Автоматическая подготовка данных для ML.

Поддерживает:
- Табличные данные (sklearn, category-encoders, imbalanced-learn)
- Текстовые данные (nlpaug, transformers)
- Изображения (albumentations, opencv)

Основные классы:
    - AutoForge: Главный класс для автоматической обработки
    - DataContainer: Контейнер данных с богатым API
    - Pipeline: Конструктор пайплайнов
    - ForgeResult: Результат обработки

Пример использования:
    >>> from automl_data import AutoForge
    >>> 
    >>> # Табличные данные
    >>> result = AutoForge(target="price").fit_transform(df)
    >>> X_train, X_test, y_train, y_test = result.get_splits()
    >>> 
    >>> # Текстовые данные
    >>> result = AutoForge(
    ...     target="sentiment",
    ...     text_column="review"
    ... ).fit_transform(df)
    >>> 
    >>> # Изображения
    >>> result = AutoForge(
    ...     target="class",
    ...     image_column="path",
    ...     image_dir="./images"
    ... ).fit_transform(df)
"""

__version__ = "1.0.0"
__author__ = "ML Data Forge Team"
__email__ = "contact@example.com"

# Core
from .core.container import (
    DataContainer,
    DataType,
    ProcessingStage,
    ProcessingStep
)
from .core.pipeline import Pipeline, PipelineResult
from .core.config import ForgeConfig, TaskType, TextConfig, ImageConfig, TabularConfig
from .core.forge import AutoForge, ForgeResult

# Adapters - Base
from .adapters.base import BaseAdapter, TransformOnlyAdapter

# Adapters - Tabular
from .adapters.profiling import ProfilerAdapter
from .adapters.encoding import EncodingAdapter
from .adapters.outliers import OutlierAdapter
from .adapters.balancing import BalancingAdapter

# Adapters - Text
from .adapters.text import TextPreprocessor, TextAugmentor

# Adapters - Image
from .adapters.image import  ImagePreprocessor, ImageAugmentor

# Utils
from .utils.exceptions import (
    MLDataForgeError,
    ValidationError,
    NotFittedError,
    PipelineError,
    ConfigurationError,
    DependencyError,
    DataQualityError
)
from .utils.dependencies import check_dependencies, print_dependency_status

# Convenience function
def forge(
    data,
    target: str = None,
    **kwargs
) -> ForgeResult:
    """
    Быстрая обработка данных в одну строку.
    
    Args:
        data: DataFrame или путь к файлу
        target: Целевая колонка
        **kwargs: Дополнительные параметры для AutoForge
    
    Returns:
        ForgeResult с обработанными данными
    
    Example:
        >>> from automl_data import forge
        >>> result = forge("data.csv", target="label")
        >>> X_train, X_test, y_train, y_test = result.get_splits()
    """
    import pandas as pd
    
    if isinstance(data, str):
        # Загружаем из файла
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(data)
        elif data.endswith('.parquet'):
            data = pd.read_parquet(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError(f"Unsupported file format: {data}")
    
    return AutoForge(target=target, **kwargs).fit_transform(data)


__all__ = [
    # Version
    "__version__",
    
    # Main
    "AutoForge",
    "ForgeResult",
    "forge",
    
    # Core
    "DataContainer",
    "DataType",
    "ProcessingStage",
    "Pipeline",
    "PipelineResult",
    
    # Config
    "ForgeConfig",
    "TaskType",
    "TextConfig",
    "ImageConfig",
    "TabularConfig",
    
    # Adapters
    "BaseAdapter",
    "ProfilerAdapter",
    "EncodingAdapter",
    "OutlierAdapter",
    "BalancingAdapter",
    "TextPreprocessor",
    "TextAugmentor",
    "ImagePreprocessor",
    "ImageAugmentor",
    
    # Exceptions
    "MLDataForgeError",
    "ValidationError",
    "NotFittedError",
    "PipelineError",
    "DependencyError",
    
    # Utils
    "check_dependencies",
    "print_dependency_status",
]