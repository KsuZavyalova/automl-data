# automl_data/utils/exceptions.py
"""
Кастомные исключения библиотеки.
"""

from __future__ import annotations

from typing import Any


class MLDataForgeError(Exception):
    """
    Базовое исключение библиотеки.
    
    Все кастомные исключения наследуются от него.
    Содержит код ошибки, детали и подсказку для исправления.
    """
    
    def __init__(
        self, 
        message: str, 
        code: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.suggestion = suggestion
    
    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.suggestion:
            parts.append(f"💡 Suggestion: {self.suggestion}")
        return "\n".join(parts)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code='{self.code}', message='{self.message}')"
    
    def to_dict(self) -> dict[str, Any]:
        """Сериализация для API/логов"""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion
        }


class ValidationError(MLDataForgeError):
    """Ошибка валидации данных"""
    
    def __init__(
        self, 
        message: str,
        column: str | None = None,
        constraint: str | None = None,
        value: Any = None,
        **kwargs
    ):
        super().__init__(
            message, 
            code="VALIDATION_ERROR",
            **kwargs
        )
        self.column = column
        self.constraint = constraint
        self.value = value
        
        if column:
            self.details["column"] = column
        if constraint:
            self.details["constraint"] = constraint
        if value is not None:
            self.details["value"] = str(value)[:100]  # Ограничиваем длину


class NotFittedError(MLDataForgeError):
    """Компонент не обучен"""
    
    def __init__(
        self, 
        message: str = "Component is not fitted",
        component: str | None = None
    ):
        super().__init__(
            message,
            code="NOT_FITTED",
            suggestion="Call fit() or fit_transform() before transform()"
        )
        self.component = component
        if component:
            self.details["component"] = component


class PipelineError(MLDataForgeError):
    """Ошибка выполнения пайплайна"""
    
    def __init__(
        self, 
        message: str,
        step: str | None = None,
        step_index: int | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(
            message, 
            code="PIPELINE_ERROR",
            suggestion="Check the failed step configuration and input data"
        )
        self.step = step
        self.step_index = step_index
        self.original_error = original_error
        
        if step:
            self.details["failed_step"] = step
        if step_index is not None:
            self.details["step_index"] = step_index
        if original_error:
            self.details["original_error"] = str(original_error)
            self.details["original_type"] = type(original_error).__name__


class ConfigurationError(MLDataForgeError):
    """Ошибка конфигурации"""
    
    def __init__(
        self, 
        message: str,
        key: str | None = None,
        expected: Any = None,
        got: Any = None
    ):
        super().__init__(
            message,
            code="CONFIG_ERROR",
            suggestion="Check your configuration parameters"
        )
        self.key = key
        if key:
            self.details["key"] = key
        if expected is not None:
            self.details["expected"] = str(expected)
        if got is not None:
            self.details["got"] = str(got)


class DependencyError(MLDataForgeError):
    """Отсутствует зависимость"""
    
    def __init__(
        self, 
        package: str, 
        install_name: str | None = None,
        feature: str | None = None
    ):
        install = install_name or package
        message = f"Required package '{package}' is not installed"
        if feature:
            message = f"Package '{package}' is required for {feature}"
        
        super().__init__(
            message,
            code="DEPENDENCY_ERROR",
            suggestion=f"Install with: pip install {install}"
        )
        self.package = package
        self.install_name = install
        self.details["package"] = package
        self.details["install_command"] = f"pip install {install}"


class DataQualityError(MLDataForgeError):
    """Ошибка качества данных"""
    
    def __init__(
        self, 
        message: str,
        quality_score: float | None = None,
        threshold: float | None = None,
        issues: list[str] | None = None
    ):
        super().__init__(
            message, 
            code="QUALITY_ERROR",
            suggestion="Review data quality issues and consider additional preprocessing"
        )
        self.quality_score = quality_score
        self.threshold = threshold
        self.issues = issues or []
        
        if quality_score is not None:
            self.details["quality_score"] = quality_score
        if threshold is not None:
            self.details["threshold"] = threshold
        if issues:
            self.details["issues"] = issues


class DataTypeError(MLDataForgeError):
    """Неподдерживаемый тип данных"""
    
    def __init__(
        self, 
        message: str,
        expected_type: str | None = None,
        actual_type: str | None = None
    ):
        super().__init__(
            message,
            code="DATA_TYPE_ERROR",
            suggestion="Check that your data matches the expected format"
        )
        if expected_type:
            self.details["expected_type"] = expected_type
        if actual_type:
            self.details["actual_type"] = actual_type


class TransformError(MLDataForgeError):
    """Ошибка трансформации"""
    
    def __init__(
        self, 
        message: str,
        transformer: str | None = None,
        column: str | None = None
    ):
        super().__init__(
            message,
            code="TRANSFORM_ERROR"
        )
        self.transformer = transformer
        self.column = column
        
        if transformer:
            self.details["transformer"] = transformer
        if column:
            self.details["column"] = column


class OutlierDetectionError(MLDataForgeError):
    """Ошибка детектирования выбросов"""
    
    def __init__(
        self,
        message: str,
        method: str | None = None,
        contamination: float | None = None
    ):
        super().__init__(
            message,
            code="OUTLIER_ERROR",
            suggestion="Check outlier detection parameters and input data"
        )
        self.method = method
        self.contamination = contamination
        
        if method:
            self.details["method"] = method
        if contamination is not None:
            self.details["contamination"] = contamination


class ImputationError(MLDataForgeError):
    """Ошибка импьютации"""
    
    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        column: str | None = None,
        missing_ratio: float | None = None
    ):
        super().__init__(
            message,
            code="IMPUTATION_ERROR",
            suggestion="Try a different imputation strategy or handle missing values manually"
        )
        self.strategy = strategy
        self.column = column
        self.missing_ratio = missing_ratio
        
        if strategy:
            self.details["strategy"] = strategy
        if column:
            self.details["column"] = column
        if missing_ratio is not None:
            self.details["missing_ratio"] = missing_ratio


class ScalingError(MLDataForgeError):
    """Ошибка масштабирования"""
    
    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        column: str | None = None
    ):
        super().__init__(
            message,
            code="SCALING_ERROR",
            suggestion="Check if data contains valid numerical values for scaling"
        )
        self.strategy = strategy
        self.column = column
        
        if strategy:
            self.details["strategy"] = strategy
        if column:
            self.details["column"] = column


class EncodingError(MLDataForgeError):
    """Ошибка кодирования"""
    
    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        column: str | None = None,
        cardinality: int | None = None
    ):
        super().__init__(
            message,
            code="ENCODING_ERROR",
            suggestion="Check categorical columns and encoding strategy"
        )
        self.strategy = strategy
        self.column = column
        self.cardinality = cardinality
        
        if strategy:
            self.details["strategy"] = strategy
        if column:
            self.details["column"] = column
        if cardinality is not None:
            self.details["cardinality"] = cardinality


class BalancingError(MLDataForgeError):
    """Ошибка балансировки классов"""
    
    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        target_column: str | None = None,
        imbalance_ratio: float | None = None
    ):
        super().__init__(
            message,
            code="BALANCING_ERROR",
            suggestion="Check class distribution and balancing parameters"
        )
        self.strategy = strategy
        self.target_column = target_column
        self.imbalance_ratio = imbalance_ratio
        
        if strategy:
            self.details["strategy"] = strategy
        if target_column:
            self.details["target_column"] = target_column
        if imbalance_ratio is not None:
            self.details["imbalance_ratio"] = imbalance_ratio