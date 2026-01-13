"""
Вспомогательные утилиты

Модули:
- dependencies: управление зависимостями
- decorators: декораторы
- exceptions: пользовательские исключения
"""

from .dependencies import (
    require_package,
    optional_import,
    check_dependencies,
    print_dependency_status,
    DependencyError,
    LazyImport
)

from .decorators import (
    singleton,
    timing,
    retry,
    timing_method, 
    require_fitted, 
    safe_transform,
    preserve_target,
    sync_container,
    CountCalls
)
from .exceptions import (
    MLDataForgeError,
    ValidationError,
    NotFittedError,
    PipelineError,
    ConfigurationError,
    DependencyError,
    DataQualityError,
    DataTypeError,
    TransformError,
    OutlierDetectionError,
    ImputationError,
    ScalingError,
    EncodingError,
    BalancingError
)

__all__ = [
    # dependencies
    'require_package',
    'optional_import',
    'check_dependencies',
    'print_dependency_status',
    'DependencyError',
    'LazyImport',
    
    # decorators
    'singleton',
    'timing',
    'retry',
    'timing_method', 
    'require_fitted', 
    'safe_transform',
    'preserve_target',
    'sync_container',
    'CountCalls',
    
    # exceptions
    'MLDataForgeError',
    'ValidationError',
    'NotFittedError',
    'PipelineError',
    'ConfigurationError',
    'DependencyError',
    'DataQualityError',
    'DataTypeError',
    'TransformError',
    'OutlierDetectionError',
    'ImputationError',
    'ScalingError',
    'EncodingError',
    'BalancingError'
    
]