"""
AutoML Data Processing Framework - Интеллектуальная подготовка данных для машинного обучения.

Пакет автоматически анализирует и готовит данные для ML:
- Автоопределение типа данных (табличные/текст/изображения)
- Автовыбор оптимальных стратегий обработки
- Производственное качество с обработкой ошибок
- Полная воспроизводимость и отчёты

Основные классы:
    AutoForge: Главный класс для автоматической обработки
    DataContainer: Универсальный контейнер данных с метаинформацией
    Pipeline: Оркестратор обработки данных
    ForgeResult: Результат обработки с методами доступа

Примеры использования:
    >>> from automl_data import AutoForge
    >>> 
    >>> # Быстрая обработка табличных данных
    >>> result = AutoForge(target="price").fit_transform(df)
    >>> X_train, X_test, y_train, y_test = result.get_splits()
    >>> 
    >>> # Текстовые данные с продвинутой обработкой
    >>> result = AutoForge(
    ...     target="sentiment",
    ...     text_column="review",
    ...     text_preprocessing_level="full"
    ... ).fit_transform(df)
    >>> 
    >>> # Изображения с аугментацией
    >>> result = AutoForge(
    ...     target="class",
    ...     image_column="path",
    ...     image_dir="./images",
    ...     augment=True
    ... ).fit_transform(df)

Лицензия: MIT
Репозиторий: https://github.com/KsuZavyalova/automl-data/tree/main
"""

__version__ = "1.0.0"
__license__ = "MIT"
__repository__ = "https://github.com/KsuZavyalova/automl-data/tree/main"

from .core.container import (
    DataContainer,
    DataType,
    ProcessingStage,
    ProcessingStep
)
from .core.pipeline import Pipeline, PipelineResult
from .core.config import ForgeConfig, TaskType, TextConfig, ImageConfig, TabularConfig
from .core.forge import AutoForge, ForgeResult

from .adapters.base import BaseAdapter, TransformOnlyAdapter

from .adapters.imputation import ImputationAdapter
from .adapters.scaling import ScalingAdapter
from .adapters.feature_cleaner import FeatureCleanerAdapter
from .adapters.encoding import EncodingAdapter
from .adapters.outliers import OutlierAdapter
from .adapters.balancing import BalancingAdapter

from .adapters.profiling import ProfilerAdapter


from .adapters.text.preprocessor import TextPreprocessor
from .adapters.text.augmentor import TextAugmentor


from .adapters.image.preprocessor import ImagePreprocessor
from .adapters.image.augmentor import ImageAugmentor

from .utils.exceptions import (
    MLDataForgeError,
    ValidationError,
    NotFittedError,
    PipelineError,
    ConfigurationError,
    DependencyError,
    DataQualityError,
    ImputationError
)
from .utils.dependencies import check_dependencies, print_dependency_status
from .utils.decorators import (
    timing,
    require_fitted,
    safe_transform,
    timing_method
)

def forge(
    data,
    target: str = None,
    **kwargs
) -> ForgeResult:
    """
    Быстрая обработка данных в одну строку.
    
    Поддерживает:
    - DataFrame, DataContainer
    - Пути к файлам: .csv, .xlsx, .parquet, .json
    - Автоопределение типа данных
    - Автовыбор стратегий обработки
    
    Args:
        data: DataFrame, DataContainer или путь к файлу
        target: Название целевой колонки (обязательно для supervised задач)
        **kwargs: Дополнительные параметры для AutoForge
    
    Returns:
        ForgeResult с обработанными данными и метаинформацией
    
    Examples:
        >>> # Из файла
        >>> result = forge("data.csv", target="price")
        >>> 
        >>> # Из DataFrame
        >>> result = forge(df, target="sentiment", text_column="review")
        >>> 
        >>> # Получение результатов
        >>> X_train, X_test, y_train, y_test = result.get_splits()
        >>> print(f"Качество данных: {result.quality_score:.0%}")
        >>> result.save_report("analysis.html")
    
    Raises:
        ValidationError: Если данные некорректны
        DependencyError: Если не хватает зависимостей
        ConfigurationError: Если конфигурация невалидна
    """
    import pandas as pd

    if isinstance(data, str):
        if data.endswith('.csv'):
            data = pd.read_csv(data)
        elif data.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(data)
        elif data.endswith('.parquet'):
            data = pd.read_parquet(data)
        elif data.endswith('.json'):
            data = pd.read_json(data)
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {data}. "
                f"Поддерживаются: .csv, .xlsx, .parquet, .json"
            )
    
    return AutoForge(target=target, **kwargs).fit_transform(data)

def get_version() -> str:
    """
    Получить версию пакета.
    
    Returns:
        Строка с версией в формате 'MAJOR.MINOR.PATCH'
    """
    return __version__


__all__ = [
    # Version and metadata
    "__version__",
    "__license__",
    "__repository__",
    
    # Main functions
    "AutoForge",
    "ForgeResult",
    "forge",
    "get_version",
    "get_citation",
    
    # Core components
    "DataContainer",
    "DataType",
    "ProcessingStage",
    "ProcessingStep",
    "Pipeline",
    "PipelineResult",
    
    # Configuration
    "ForgeConfig",
    "TaskType",
    "TextConfig",
    "ImageConfig",
    "TabularConfig",
    
    # Base adapters
    "BaseAdapter",
    "TransformOnlyAdapter",
    
    # Tabular adapters
    "ProfilerAdapter",
    "FeatureCleanerAdapter",
    "ImputationAdapter",
    "ScalingAdapter",
    "EncodingAdapter",
    "OutlierAdapter",
    "BalancingAdapter",
    
    # Text adapters
    "TextPreprocessor",
    "TextAugmentor",
    
    # Image adapters
    "ImagePreprocessor",
    "ImageAugmentor",
    
    # Exceptions
    "MLDataForgeError",
    "ValidationError",
    "NotFittedError",
    "PipelineError",
    "ConfigurationError",
    "DependencyError",
    "DataQualityError",
    "ImputationError",
    
    # Utilities
    "check_dependencies",
    "print_dependency_status",
    "timing",
    "require_fitted",
    "safe_transform",
    "timing_method",
]


try:
    import pandas as pd
    import numpy as np
    MIN_PANDAS_VERSION = "1.3.0"
    MIN_NUMPY_VERSION = "1.21.0"
    
    from packaging import version
    
    if version.parse(pd.__version__) < version.parse(MIN_PANDAS_VERSION):
        raise ImportError(
            f"pandas >= {MIN_PANDAS_VERSION} required, "
            f"got {pd.__version__}"
        )
    
    if version.parse(np.__version__) < version.parse(MIN_NUMPY_VERSION):
        raise ImportError(
            f"numpy >= {MIN_NUMPY_VERSION} required, "
            f"got {np.__version__}"
        )
    
except ImportError as e:
    raise ImportError(
        f"AutoML Data requires pandas>={MIN_PANDAS_VERSION} and "
        f"numpy>={MIN_NUMPY_VERSION}. Please install with: "
        f"pip install 'pandas>={MIN_PANDAS_VERSION}' 'numpy>={MIN_NUMPY_VERSION}'"
    ) from e

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

_initialized_message = f"""
{'='*60}
AutoML Data Processing Framework v{__version__}
{'='*60}
✅ Successfully imported!

Quick start:
    >>> from automl_data import forge
    >>> result = forge("your_data.csv", target="target_column")
    >>> X_train, X_test, y_train, y_test = result.get_splits()

Repository: {__repository__}
{'='*60}
"""

import sys
if hasattr(sys, 'ps1') and not any('jupyter' in x for x in sys.modules):
    print(_initialized_message)