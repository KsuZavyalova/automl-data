"""
Тесты кастомных исключений библиотеки.
"""

from __future__ import annotations

import pytest
from automl_data.utils.exceptions import (
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


class TestMLDataForgeError:
    """Тесты базового исключения"""
    
    def test_basic_error(self):
        """Базовое исключение с сообщением"""
        error = MLDataForgeError("Test error")
        
        assert str(error) == "[UNKNOWN_ERROR] Test error"
        assert error.code == "UNKNOWN_ERROR"
        assert error.message == "Test error"
        assert error.details == {}
        assert error.suggestion is None
    
    def test_error_with_suggestion(self):
        """Исключение с подсказкой"""
        error = MLDataForgeError(
            message="Test error",
            code="CUSTOM_CODE",
            suggestion="Try this fix"
        )
        
        assert error.code == "CUSTOM_CODE"
        assert error.suggestion == "Try this fix"
        assert "💡 Suggestion: Try this fix" in str(error)
    
    def test_error_with_details(self):
        """Исключение с деталями"""
        error = MLDataForgeError(
            message="Test error",
            details={"key": "value", "count": 42}
        )
        
        assert error.details["key"] == "value"
        assert error.details["count"] == 42
    
    def test_to_dict(self):
        """Сериализация в словарь"""
        error = MLDataForgeError(
            message="Test error",
            code="TEST",
            details={"test": "data"},
            suggestion="Fix it"
        )
        
        result = error.to_dict()
        
        assert result["error_type"] == "MLDataForgeError"
        assert result["code"] == "TEST"
        assert result["message"] == "Test error"
        assert result["details"] == {"test": "data"}
        assert result["suggestion"] == "Fix it"
    
    def test_repr(self):
        """Тест repr"""
        error = MLDataForgeError("Test error", code="TEST")
        
        assert repr(error) == "MLDataForgeError(code='TEST', message='Test error')"


class TestValidationError:
    """Тесты ValidationError"""
    
    def test_validation_error_basic(self):
        """Базовая валидационная ошибка"""
        error = ValidationError("Invalid data")
        
        assert error.code == "VALIDATION_ERROR"
        assert error.column is None
        assert error.constraint is None
        assert error.value is None
    
    def test_validation_error_with_column(self):
        """Валидационная ошибка с колонкой"""
        error = ValidationError(
            message="Column has invalid values",
            column="age",
            constraint="> 0",
            value=-5
        )
        
        assert error.column == "age"
        assert error.constraint == "> 0"
        assert error.value == -5
        assert error.details["column"] == "age"
        assert error.details["constraint"] == "> 0"
        assert error.details["value"] == "-5"
    



class TestNotFittedError:
    """Тесты NotFittedError"""
    
    def test_not_fitted_error_default(self):
        """Ошибка по умолчанию"""
        error = NotFittedError()
        
        assert error.code == "NOT_FITTED"
        assert error.message == "Component is not fitted"
        assert "fit() or fit_transform()" in error.suggestion
        assert error.component is None
    
    def test_not_fitted_error_with_component(self):
        """Ошибка с указанием компонента"""
        error = NotFittedError(
            message="Scaler is not fitted",
            component="StandardScaler"
        )
        
        assert error.component == "StandardScaler"
        assert error.details["component"] == "StandardScaler"


class TestPipelineError:
    """Тесты PipelineError"""
    
    def test_pipeline_error_basic(self):
        """Базовая ошибка пайплайна"""
        error = PipelineError("Pipeline failed")
        
        assert error.code == "PIPELINE_ERROR"
        assert error.step is None
        assert error.step_index is None
        assert error.original_error is None
    
    def test_pipeline_error_with_step(self):
        """Ошибка пайплайна с шагом"""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            original = e
        
        error = PipelineError(
            message="Step failed",
            step="imputation",
            step_index=2,
            original_error=original
        )
        
        assert error.step == "imputation"
        assert error.step_index == 2
        assert error.original_error == original
        assert error.details["failed_step"] == "imputation"
        assert error.details["step_index"] == 2
        assert error.details["original_type"] == "ValueError"


class TestConfigurationError:
    """Тесты ConfigurationError"""
    
    def test_configuration_error(self):
        """Ошибка конфигурации"""
        error = ConfigurationError(
            message="Invalid parameter",
            key="n_estimators",
            expected="int > 0",
            got=-5
        )
        
        assert error.code == "CONFIG_ERROR"
        assert error.key == "n_estimators"
        assert error.details["key"] == "n_estimators"
        assert error.details["expected"] == "int > 0"
        assert error.details["got"] == "-5"


class TestDependencyError:
    """Тесты DependencyError"""
    
    def test_dependency_error(self):
        """Ошибка зависимости"""
        error = DependencyError(
            package="scikit-learn",
            install_name="scikit-learn",
            feature="machine learning algorithms"
        )
        
        assert error.code == "DEPENDENCY_ERROR"
        assert error.package == "scikit-learn"
        assert "scikit-learn" in error.message
        assert "pip install scikit-learn" in error.suggestion
        assert error.details["install_command"] == "pip install scikit-learn"


class TestDataQualityError:
    """Тесты DataQualityError"""
    
    def test_data_quality_error(self):
        """Ошибка качества данных"""
        issues = ["Missing values: 10%", "Outliers detected"]
        error = DataQualityError(
            message="Low data quality",
            quality_score=0.6,
            threshold=0.8,
            issues=issues
        )
        
        assert error.code == "QUALITY_ERROR"
        assert error.quality_score == 0.6
        assert error.threshold == 0.8
        assert error.issues == issues
        assert error.details["quality_score"] == 0.6
        assert error.details["threshold"] == 0.8
        assert error.details["issues"] == issues


class TestDataTypeError:
    """Тесты DataTypeError"""
    
    def test_data_type_error(self):
        """Ошибка типа данных"""
        error = DataTypeError(
            message="Expected DataFrame",
            expected_type="pandas.DataFrame",
            actual_type="list"
        )
        
        assert error.code == "DATA_TYPE_ERROR"
        assert error.details["expected_type"] == "pandas.DataFrame"
        assert error.details["actual_type"] == "list"


class TestTransformError:
    """Тесты TransformError"""
    
    def test_transform_error(self):
        """Ошибка трансформации"""
        error = TransformError(
            message="Cannot transform column",
            transformer="OneHotEncoder",
            column="category"
        )
        
        assert error.code == "TRANSFORM_ERROR"
        assert error.transformer == "OneHotEncoder"
        assert error.column == "category"
        assert error.details["transformer"] == "OneHotEncoder"
        assert error.details["column"] == "category"


class TestAdapterSpecificErrors:
    """Тесты специфичных исключений адаптеров"""
    
    def test_outlier_detection_error(self):
        """Ошибка детектирования выбросов"""
        error = OutlierDetectionError(
            message="Failed to detect outliers",
            method="IsolationForest",
            contamination=0.1
        )
        
        assert error.code == "OUTLIER_ERROR"
        assert error.method == "IsolationForest"
        assert error.contamination == 0.1
        assert error.details["method"] == "IsolationForest"
        assert error.details["contamination"] == 0.1
    
    def test_imputation_error(self):
        """Ошибка импьютации"""
        error = ImputationError(
            message="Cannot impute column",
            strategy="knn",
            column="age",
            missing_ratio=0.5
        )
        
        assert error.code == "IMPUTATION_ERROR"
        assert error.strategy == "knn"
        assert error.column == "age"
        assert error.missing_ratio == 0.5
        assert error.details["strategy"] == "knn"
        assert error.details["column"] == "age"
        assert error.details["missing_ratio"] == 0.5
    
    def test_scaling_error(self):
        """Ошибка масштабирования"""
        error = ScalingError(
            message="Cannot scale constant column",
            strategy="StandardScaler",
            column="constant_feature"
        )
        
        assert error.code == "SCALING_ERROR"
        assert error.strategy == "StandardScaler"
        assert error.column == "constant_feature"
        assert error.details["strategy"] == "StandardScaler"
        assert error.details["column"] == "constant_feature"
    
    def test_encoding_error(self):
        """Ошибка кодирования"""
        error = EncodingError(
            message="High cardinality column",
            strategy="OneHotEncoder",
            column="user_id",
            cardinality=10000
        )
        
        assert error.code == "ENCODING_ERROR"
        assert error.strategy == "OneHotEncoder"
        assert error.column == "user_id"
        assert error.cardinality == 10000
        assert error.details["strategy"] == "OneHotEncoder"
        assert error.details["column"] == "user_id"
        assert error.details["cardinality"] == 10000
    
    def test_balancing_error(self):
        """Ошибка балансировки"""
        error = BalancingError(
            message="Cannot balance data",
            strategy="SMOTE",
            target_column="target",
            imbalance_ratio=0.1
        )
        
        assert error.code == "BALANCING_ERROR"
        assert error.strategy == "SMOTE"
        assert error.target_column == "target"
        assert error.imbalance_ratio == 0.1
        assert error.details["strategy"] == "SMOTE"
        assert error.details["target_column"] == "target"
        assert error.details["imbalance_ratio"] == 0.1


class TestExceptionHierarchy:
    """Тесты иерархии исключений"""
    
    def test_all_errors_inherit_from_base(self):
        """Все кастомные исключения наследуются от MLDataForgeError"""
        error_classes = [
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
        ]
        
        for error_class in error_classes:
            instance = error_class("Test")
            assert isinstance(instance, MLDataForgeError)
            assert issubclass(error_class, MLDataForgeError)
    
    def test_exception_chaining(self):
        """Цепочка исключений"""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            original = e
        
        try:
            raise PipelineError(
                "Pipeline failed",
                step="imputation",
                original_error=original
            )
        except PipelineError as e:
            pipeline_error = e
        
        assert isinstance(pipeline_error.original_error, ValueError)
        assert str(original) in str(pipeline_error.details["original_error"])


class TestExceptionUsage:
    """Тесты использования исключений"""
    
    def test_raise_and_catch(self):
        """Поднятие и перехват исключений"""
        try:
            raise ValidationError("Invalid data", column="age")
        except ValidationError as e:
            caught = e
        
        assert caught.column == "age"
        assert isinstance(caught, MLDataForgeError)
    
    def test_error_inheritance_catch(self):
        """Перехват через базовый класс"""
        try:
            raise NotFittedError("Component not fitted")
        except MLDataForgeError as e:
            caught = e
        
        assert isinstance(caught, NotFittedError)
        assert caught.code == "NOT_FITTED"
    
