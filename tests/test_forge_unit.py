"""
Юнит-тесты для AutoForge и ForgeResult.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import warnings
from unittest.mock import Mock, patch, MagicMock

from automl_data.core.forge import AutoForge, ForgeResult
from automl_data.core.container import DataContainer, DataType, ProcessingStage
from automl_data.core.config import TaskType
from automl_data.core.pipeline import Pipeline


# ==================== FIXTURES ====================

@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Простой датафрейм для тестов"""
    np.random.seed(42)
    return pd.DataFrame({
        'num_feature': np.random.randn(100),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Данные для регрессии"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)  # Непрерывная цель
    })


@pytest.fixture
def text_df() -> pd.DataFrame:
    """Текстовые данные"""
    np.random.seed(42)
    texts = [
        "This is a positive review about the product",
        "Negative experience with customer service",
        "Average product, nothing special",
        "Excellent quality and fast delivery",
        "Would not recommend to anyone"
    ] * 20
    return pd.DataFrame({
        'text': texts,
        'sentiment': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """Несбалансированные данные"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': [0] * 90 + [1] * 10  # 90% класса 0, 10% класса 1
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Данные с пропусками"""
    np.random.seed(42)
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    # Добавляем пропуски
    df.loc[10:20, 'num1'] = np.nan
    df.loc[30:35, 'cat'] = None
    return df


# ==================== FORGE RESULT TESTS ====================

class TestForgeResult:
    """Тесты ForgeResult"""
    
    def test_forge_result_basic_properties(self, simple_df):
        """Базовые свойства ForgeResult"""
        container = DataContainer(data=simple_df, target_column='target')
        result = ForgeResult(
            container=container,
            config=Mock(),
            execution_time=1.5
        )
        
        assert result.data.shape == (100, 3)
        assert result.X.shape == (100, 2)
        assert result.y.shape == (100,)
        assert result.quality_score == 1.0
        assert result.shape == (100, 3)
        assert result.steps == []
        assert result.execution_time == 1.5
    
    def test_get_splits_basic(self, simple_df):
        """Разделение на train/test"""
        container = DataContainer(data=simple_df, target_column='target')
        
        # Мокаем config с нужными атрибутами
        mock_config = Mock()
        mock_config.test_size = 0.3
        mock_config.random_state = 42
        mock_config.stratify = True
        
        result = ForgeResult(container=container, config=mock_config)
        
        X_train, X_test, y_train, y_test = result.get_splits()
        
        assert len(X_train) == 70  # 70% от 100
        assert len(X_test) == 30   # 30% от 100
        assert X_train.shape[1] == 2  # 2 признака
        assert y_train.shape == (70,)
        assert y_test.shape == (30,)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
    
    def test_get_splits_without_stratify(self, simple_df):
        """Разделение без стратификации"""
        container = DataContainer(data=simple_df, target_column='target')
        
        mock_config = Mock()
        mock_config.test_size = 0.2
        mock_config.random_state = 42
        mock_config.stratify = False
        
        result = ForgeResult(container=container, config=mock_config)
        
        X_train, X_test, y_train, y_test = result.get_splits()
        
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_get_splits_custom_params(self, simple_df):
        """Разделение с пользовательскими параметрами"""
        container = DataContainer(data=simple_df, target_column='target')
        
        mock_config = Mock()
        mock_config.test_size = 0.2
        mock_config.random_state = 42
        mock_config.stratify = True
        
        result = ForgeResult(container=container, config=mock_config)
        
        X_train, X_test, y_train, y_test = result.get_splits(
            test_size=0.25,
            random_state=123,
            stratify=False
        )
        
        assert len(X_train) == 75  # 75% от 100
        assert len(X_test) == 25   # 25% от 100
    
    def test_get_splits_no_target_error(self, simple_df):
        """Ошибка при отсутствии target"""
        container = DataContainer(data=simple_df)  # Без target
        result = ForgeResult(container=container, config=Mock())
        
        with pytest.raises(ValueError, match="Target column not specified or not found"):
            result.get_splits()
    
    def test_to_numpy(self, simple_df):
        """Конвертация в numpy"""
        container = DataContainer(data=simple_df, target_column='target')
        result = ForgeResult(container=container, config=Mock())
        
        X_np, y_np = result.to_numpy()
        
        assert isinstance(X_np, np.ndarray)
        assert isinstance(y_np, np.ndarray)
        assert X_np.shape == (100, 2)
        assert y_np.shape == (100,)
    
    def test_to_numpy_no_target(self, simple_df):
        """Конвертация в numpy без target"""
        container = DataContainer(data=simple_df)  # Без target
        result = ForgeResult(container=container, config=Mock())
        
        X_np, y_np = result.to_numpy()
        
        assert X_np.shape == (100, 3)  # Все колонки
        assert y_np is None
    
    def test_save_report_no_report(self, simple_df, tmp_path):
        """Сохранение отчёта, когда его нет"""
        container = DataContainer(data=simple_df, target_column='target')
        result = ForgeResult(container=container, config=Mock())
        
        # Не должно падать, просто ничего не делает
        result.save_report(tmp_path / "report.html")
    
    def test_summary(self, simple_df):
        """Сводка результата"""
        # Создаем контейнер с реальными значениями
        container = DataContainer(
            data=simple_df,
            target_column='target'
        )
        container.data_type = DataType.TABULAR
        container.stage = ProcessingStage.TRANSFORMED
        
        # Добавляем шаги обработки
        from automl_data.core.container import ProcessingStep
        container.processing_history = [
            ProcessingStep(name="step1", component="Adapter1"),
            ProcessingStep(name="step2", component="Adapter2")
        ]
        
        # Создаем mock конфига с методом to_dict
        mock_config = Mock()
        mock_config.to_dict = Mock(return_value={"test_size": 0.2})
        
        result = ForgeResult(
            container=container,
            config=mock_config,
            execution_time=2.5
        )
        
        summary = result.summary()
        
        assert "shape" in summary
        assert "quality" in summary
        assert "steps" in summary
        assert "execution_time" in summary
        assert "config" in summary
        assert summary["execution_time"] == "2.50s"
    
    def test_repr(self, simple_df):
        """Тест repr"""
        container = DataContainer(data=simple_df, target_column='target')
        container.quality_score = 0.85
        
        result = ForgeResult(
            container=container,
            config=Mock(),
            execution_time=1.23
        )
        
        repr_str = repr(result)
        
        assert "ForgeResult" in repr_str
        assert "shape=(100, 3)" in repr_str
        assert "quality=85%" in repr_str
        assert "time=1.23s" in repr_str


# ==================== AUTO FORGE INITIALIZATION TESTS ====================

class TestAutoForgeInitialization:
    """Тесты инициализации AutoForge"""
    
    def test_basic_initialization(self):
        """Базовая инициализация"""
        forge = AutoForge(target="price")
        
        assert forge.config.target == "price"
        assert forge.config.task == TaskType.AUTO
        assert forge.config.verbose is True
        assert forge.config.random_state == 42
        assert forge._is_fitted is False
        assert forge._pipeline is None
    
    def test_initialization_with_all_params(self):
        """Инициализация со всеми параметрами"""
        forge = AutoForge(
            target="sentiment",
            task="classification",
            text_column="review",
            impute_strategy="knn",
            scaling="standard",
            encode_strategy="onehot",
            outlier_method="iforest",
            balance=False,
            test_size=0.3,
            random_state=123,
            verbose=False
        )
        
        assert forge.config.target == "sentiment"
        assert forge.config.task == TaskType.CLASSIFICATION
        assert forge.text_column == "review"
        assert forge.config.tabular.impute_strategy == "knn"
        assert forge.config.tabular.scaling == "standard"
        assert forge.config.tabular.encode_strategy == "onehot"
        assert forge.config.tabular.outlier_method == "iforest"
        assert forge.config.balance is False
        assert forge.config.test_size == 0.3
        assert forge.config.random_state == 123
        assert forge.config.verbose is False
    
    def test_initialization_with_tabular_config(self):
        """Инициализация с TabularConfig"""
        from automl_data.core.config import TabularConfig
        
        tabular_config = TabularConfig(
            impute_strategy="iterative",
            scaling="robust",
            encode_strategy="target",
            max_onehot_cardinality=15
        )
        
        forge = AutoForge(
            target="target",
            tabular_config=tabular_config
        )
        
        assert forge.config.tabular.impute_strategy == "iterative"
        assert forge.config.tabular.scaling == "robust"
        assert forge.config.tabular.encode_strategy == "target"
        assert forge.config.tabular.max_onehot_cardinality == 15
    
    def test_initialization_with_text_config(self):
        """Инициализация с TextConfig"""
        from automl_data.core.config import TextConfig
        
        text_config = TextConfig(
            preprocessing_level="full",  # Изменено с "advanced" на "full"
            lowercase=False,
            lemmatize=False,
            augment=True,
            augment_factor=2.5
        )
        
        forge = AutoForge(
            target="sentiment",
            text_column="review",
            text_config=text_config
        )
        
        assert forge.config.text.preprocessing_level == "full"
        assert forge.config.text.lowercase is False
        assert forge.config.text.lemmatize is False
        assert forge.config.text.augment is True
        assert forge.config.text.augment_factor == 2.5
    
    def test_initialization_with_image_config(self):
        """Инициализация с ImageConfig"""
        from automl_data.core.config import ImageConfig
        
        image_config = ImageConfig(
            augment=True,
            augment_factor=3.0,
            target_size=(224, 224),
            rotation_range=30
        )
        
        forge = AutoForge(
            target="class",
            image_column="path",
            image_config=image_config
        )
        
        assert forge.config.image.augment is True
        assert forge.config.image.augment_factor == 3.0
        assert forge.config.image.target_size == (224, 224)
        assert forge.config.image.rotation_range == 30
    
    def test_repr_not_fitted(self):
        """Тест repr для необученного объекта"""
        forge = AutoForge(target="target")
        
        repr_str = repr(forge)
        
        assert "AutoForge" in repr_str
        assert "target='target'" in repr_str
        assert "task=auto" in repr_str
        assert "not fitted" in repr_str
    
    def test_repr_fitted(self, simple_df):
        """Тест repr для обученного объекта"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с атрибутом len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
        
        repr_str = repr(forge)
        
        assert "AutoForge" in repr_str
        assert "target='target'" in repr_str
        assert "fitted" in repr_str


# ==================== AUTO FORGE FIT TESTS ====================

class TestAutoForgeFit:
    """Тесты метода fit"""
    
    def test_fit_with_dataframe(self, simple_df):
        """Обучение с DataFrame"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            result = forge.fit(simple_df)
        
        assert result is forge  # Возвращает self
        assert forge._is_fitted is True
        assert forge._pipeline is not None
        assert forge._data_type == DataType.TABULAR
    
    def test_fit_with_container(self, simple_df):
        """Обучение с DataContainer"""
        container = DataContainer(data=simple_df, target_column="target")
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            result = forge.fit(container)
        
        assert result is forge
        assert forge._is_fitted is True
    
    def test_fit_detects_regression(self, regression_df):
        """Определение задачи регрессии"""
        forge = AutoForge(target="target", verbose=False)
        
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(regression_df)
        
        assert forge.config.task == TaskType.REGRESSION
    
    def test_fit_detects_classification(self, simple_df):
        """Определение задачи классификации"""
        forge = AutoForge(target="target", verbose=False)
        
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
        
        assert forge.config.task == TaskType.CLASSIFICATION
    
    def test_fit_with_explicit_task(self, regression_df):
        """Явное указание задачи"""
        forge = AutoForge(target="target", task="classification", verbose=False)
        
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(regression_df)
        
        # Должен использовать явно указанную задачу, а не автоопределение
        assert forge.config.task == TaskType.CLASSIFICATION
    
    def test_fit_with_missing_target_error(self, simple_df):
        """Ошибка при отсутствии целевой колонки"""
        forge = AutoForge(target="non_existent", verbose=False)
        
        with pytest.raises(Exception, match="Target column.*not found"):
            mock_pipeline = Mock()
            mock_pipeline.__len__ = Mock(return_value=5)
            with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
                forge.fit(simple_df)
    
    def test_fit_empty_data_error(self):
        """Ошибка при пустых данных"""
        forge = AutoForge(target="target", verbose=False)
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception, match="Data is empty"):
            mock_pipeline = Mock()
            mock_pipeline.__len__ = Mock(return_value=5)
            with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
                forge.fit(empty_df)
    
    def test_fit_with_text_data(self, text_df):
        """Обучение с текстовыми данными"""
        forge = AutoForge(target="sentiment", text_column="text", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_text_pipeline', return_value=mock_pipeline):
            forge.fit(text_df)
        
        assert forge._data_type == DataType.TEXT
    
    def test_fit_updates_imbalance_threshold(self, simple_df):
        """Проверка обновления порога дисбаланса"""
        forge = AutoForge(target="target", balance_threshold=0.4, verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
        
        # Проверяем, что порог установлен
        assert hasattr(forge.config, 'balance_threshold')
        assert forge.config.balance_threshold == 0.4
    
    def test_fit_logs_info(self, simple_df, caplog):
        """Логирование при обучении"""
        import logging
        caplog.set_level(logging.INFO)
        
        forge = AutoForge(target="target", verbose=True)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            # Мокаем ProfilerAdapter чтобы избежать реального профилирования
            with patch('automl_data.core.forge.ProfilerAdapter') as MockProfiler:
                mock_instance = Mock()
                mock_instance.transform.return_value = Mock()
                MockProfiler.return_value = mock_instance
                
                forge.fit(simple_df)
        
        # Проверяем логи
        assert "Analyzing data" in caplog.text
        assert "Data type" in caplog.text
        assert "Pipeline ready" in caplog.text
    
    def test_fit_creates_profiler(self, simple_df):
        """Создание профилировщика при verbose=True"""
        forge = AutoForge(target="target", verbose=True)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            with patch('automl_data.core.forge.ProfilerAdapter') as MockProfiler:
                mock_instance = Mock()
                mock_instance.transform.return_value = Mock()
                MockProfiler.return_value = mock_instance
                
                forge.fit(simple_df)
        
        assert forge._profiler is not None
    
    def test_fit_without_verbose_no_profiler(self, simple_df):
        """Без verbose не создаётся профилировщик"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
        
        assert forge._profiler is None


# ==================== AUTO FORGE TRANSFORM TESTS ====================

class TestAutoForgeTransform:
    """Тесты метода transform"""
    
    def test_transform_not_fitted_error(self, simple_df):
        """Ошибка при transform до fit"""
        forge = AutoForge(target="target")
        
        with pytest.raises(Exception, match="AutoForge is not fitted"):
            forge.transform(simple_df)
    
    def test_fit_transform_basic(self, simple_df):
        """Базовая fit_transform"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=True,
            errors=[],
            container=DataContainer(data=simple_df, target_column="target")
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            result = forge.fit_transform(simple_df)
        
        assert isinstance(result, ForgeResult)
        # DataContainer может добавлять служебные колонки
        assert len(result.container.data) == 100
        assert forge._is_fitted is True
    
    def test_transform_with_dataframe(self, simple_df):
        """Transform с DataFrame"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len и execute
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=True,
            errors=[],
            container=DataContainer(data=simple_df, target_column="target")
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)  # Сначала fit
            result = forge.transform(simple_df)  # Затем transform
        
        assert isinstance(result, ForgeResult)
    
    def test_transform_with_container(self, simple_df):
        """Transform с DataContainer"""
        container = DataContainer(data=simple_df, target_column="target")
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len и execute
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=True,
            errors=[],
            container=container
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(container)
            result = forge.transform(container)
        
        assert isinstance(result, ForgeResult)
    
    def test_transform_with_pipeline_errors(self, simple_df):
        """Transform с ошибками в пайплайне"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=False,
            errors=["Error in step 1", "Error in step 2"],
            container=DataContainer(data=simple_df, target_column="target")
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
            
            # Не должен падать, но должен логировать ошибки
            with patch.object(forge, '_log') as mock_log:
                result = forge.transform(simple_df)
                # Проверяем, что _log вызывался с любыми аргументами
                assert mock_log.called
        
        assert isinstance(result, ForgeResult)
    
    def test_transform_decorators(self, simple_df):
        """Проверка декораторов transform"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len и execute
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=True,
            errors=[],
            container=DataContainer(data=simple_df, target_column="target")
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(simple_df)
            
            # Мокаем декоратор timing
            with patch('automl_data.core.forge.timing', lambda f: f):
                # Просто проверяем, что transform работает
                result = forge.transform(simple_df)
        
        assert isinstance(result, ForgeResult)


# ==================== AUTO FORGE PIPELINE BUILDING TESTS ====================

class TestAutoForgePipelineBuilding:
    """Тесты построения пайплайнов"""
    
    def test_build_tabular_pipeline_basic(self, simple_df):
        """Построение табличного пайплайна"""
        forge = AutoForge(target="target", verbose=False)
        container = DataContainer(data=simple_df, target_column="target")
        
        pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
        # В табличном пайплайне должны быть шаги очистки, импьютации и т.д.
        # Так как это реальный вызов, pipeline будет реальным объектом Pipeline
    
    def test_build_tabular_pipeline_with_balance(self, imbalanced_df):
        """Табличный пайплайн с балансировкой"""
        forge = AutoForge(target="target", balance=True, verbose=False)
        container = DataContainer(data=imbalanced_df, target_column="target")
        
        # Мокаем определение задачи как классификации
        with patch.object(forge, '_infer_task', return_value=TaskType.CLASSIFICATION):
            pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
    
    def test_build_tabular_pipeline_no_balance_for_regression(self, regression_df):
        """Нет балансировки для регрессии"""
        forge = AutoForge(target="target", balance=True, verbose=False)
        container = DataContainer(data=regression_df, target_column="target")
        
        # Мокаем определение задачи как регрессии
        with patch.object(forge, '_infer_task', return_value=TaskType.REGRESSION):
            pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
    
    def test_build_tabular_pipeline_no_outliers(self, simple_df):
        """Табличный пайплайн без обработки выбросов"""
        forge = AutoForge(
            target="target",
            outlier_method="none",
            verbose=False
        )
        container = DataContainer(data=simple_df, target_column="target")
        
        pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
    
    def test_build_tabular_pipeline_no_scaling(self, simple_df):
        """Табличный пайплайн без масштабирования"""
        forge = AutoForge(
            target="target",
            scaling="none",
            verbose=False
        )
        container = DataContainer(data=simple_df, target_column="target")
        
        pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
    
    def test_build_tabular_pipeline_no_categorical_columns(self, regression_df):
        """Табличный пайплайн без категориальных колонок"""
        forge = AutoForge(target="target", verbose=False)
        container = DataContainer(data=regression_df, target_column="target")
        
        pipeline = forge._build_pipeline(container)
        
        assert pipeline is not None
    
    def test_build_text_pipeline_mocked(self):
        """Мок построения текстового пайплайна"""
        forge = AutoForge(target="sentiment", text_column="text", verbose=False)
        pipeline_mock = Mock()
        container_mock = Mock()
        container_mock.is_imbalanced = False
        container_mock.data_type = DataType.TEXT
        
        # Создаем реальный pipeline
        real_pipeline = Pipeline(name="test")
        
        # Мокаем импорт текстовых адаптеров
        with patch.dict('sys.modules', {
            'automl_data.adapters.text': Mock(
                TextPreprocessor=Mock(),
                TextAugmentor=Mock()
            )
        }):
            with patch('automl_data.core.forge.Pipeline', return_value=real_pipeline):
                result = forge._build_text_pipeline(real_pipeline, container_mock)
        
        assert result is real_pipeline
    
    def test_build_text_pipeline_with_augmentation(self):
        """Текстовый пайплайн с аугментацией"""
        forge = AutoForge(
            target="sentiment",
            text_column="text",
            text_augment=True,
            text_augment_factor=2.0,
            verbose=False
        )
        
        # Создаем реальный pipeline
        real_pipeline = Pipeline(name="test")
        container_mock = Mock()
        container_mock.is_imbalanced = False
        container_mock.data_type = DataType.TEXT
        
        with patch.dict('sys.modules', {
            'automl_data.adapters.text': Mock(
                TextPreprocessor=Mock(),
                TextAugmentor=Mock()
            )
        }):
            with patch('automl_data.core.forge.Pipeline', return_value=real_pipeline):
                result = forge._build_text_pipeline(real_pipeline, container_mock)
        
        assert result is real_pipeline
    
    def test_build_image_pipeline_mocked(self):
        """Мок построения пайплайна для изображений"""
        forge = AutoForge(target="class", image_column="path", verbose=False)
        
        # Создаем реальный pipeline
        real_pipeline = Pipeline(name="test")
        container_mock = Mock()
        container_mock.is_imbalanced = False
        container_mock.data_type = DataType.IMAGE
        
        # Мокаем импорт адаптеров изображений
        with patch.dict('sys.modules', {
            'automl_data.adapters.image': Mock(
                ImagePreprocessor=Mock(),
                ImageAugmentor=Mock()
            )
        }):
            with patch('automl_data.core.forge.Pipeline', return_value=real_pipeline):
                result = forge._build_image_pipeline(real_pipeline, container_mock)
        
        assert result is real_pipeline


# ==================== AUTO FORGE UTILITY TESTS ====================

class TestAutoForgeUtilities:
    """Тесты вспомогательных методов"""
    
    def test_validate_input_valid(self, simple_df):
        """Валидация корректных данных"""
        forge = AutoForge(target="target")
        container = DataContainer(data=simple_df, target_column="target")
        
        # Не должно вызывать исключений
        forge._validate_input(container)
    
    def test_validate_input_empty(self):
        """Валидация пустых данных"""
        forge = AutoForge(target="target")
        container = DataContainer(data=pd.DataFrame(), target_column="target")
        
        with pytest.raises(Exception, match="Data is empty"):
            forge._validate_input(container)
    
    def test_validate_input_missing_target(self, simple_df):
        """Валидация при отсутствующем target"""
        forge = AutoForge(target="non_existent")
        container = DataContainer(data=simple_df, target_column="target")
        
        with pytest.raises(Exception, match="Target column.*not found"):
            forge._validate_input(container)
    
    def test_infer_task_classification(self, simple_df):
        """Определение задачи классификации"""
        forge = AutoForge(target="target")
        container = DataContainer(data=simple_df, target_column="target")
        
        task = forge._infer_task(container)
        
        assert task == TaskType.CLASSIFICATION
    
    def test_infer_task_regression(self, regression_df):
        """Определение задачи регрессии"""
        forge = AutoForge(target="target")
        container = DataContainer(data=regression_df, target_column="target")
        
        task = forge._infer_task(container)
        
        assert task == TaskType.REGRESSION
    
    def test_calculate_quality_basic(self, simple_df):
        """Расчёт качества данных"""
        forge = AutoForge(target="target")
        container = DataContainer(data=simple_df, target_column="target")
        
        quality = forge._calculate_quality(container)
        
        assert 0.0 <= quality <= 1.0
        assert isinstance(quality, float)
    
    def test_calculate_quality_with_missing(self, df_with_missing):
        """Расчёт качества с пропусками"""
        forge = AutoForge(target="target")
        container = DataContainer(data=df_with_missing, target_column="target")
        
        quality = forge._calculate_quality(container)
        
        assert 0.0 <= quality <= 1.0
        # Качество должно быть ниже, чем в данных без пропусков
        quality_no_missing = forge._calculate_quality(
            DataContainer(data=df_with_missing.fillna(0), target_column="target")
        )
        assert quality <= quality_no_missing
    
    def test_calculate_quality_imbalanced(self, imbalanced_df):
        """Расчёт качества для несбалансированных данных"""
        forge = AutoForge(target="target")
        container = DataContainer(data=imbalanced_df, target_column="target")
        
        quality = forge._calculate_quality(container)
        
        assert 0.0 <= quality <= 1.0
    
    def test_log_verbose(self, caplog):
        """Логирование при verbose=True"""
        import logging
        caplog.set_level(logging.INFO)
        
        forge = AutoForge(target="target", verbose=True)
        forge._log("Test message")
        
        assert "Test message" in caplog.text
    
    def test_log_not_verbose(self, caplog):
        """Отсутствие логирования при verbose=False"""
        import logging
        caplog.set_level(logging.INFO)
        
        forge = AutoForge(target="target", verbose=False)
        forge._log("Test message")
        
        assert "Test message" not in caplog.text


# ==================== EDGE CASES AND INTEGRATION TESTS ====================

class TestAutoForgeEdgeCases:
    """Тесты граничных случаев"""
    
    def test_single_row_dataframe(self):
        """Обработка датафрейма с одной строкой"""
        df = pd.DataFrame({
            'feature': [1.0],
            'target': [0]
        })
        
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        # Не должно падать
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(df)
        
        assert forge._is_fitted is True
    
    def test_all_numeric_columns(self):
        """Все колонки числовые"""
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'num3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(df)
        
        assert forge._is_fitted is True
    
    def test_high_cardinality_categorical(self):
        """Высокая кардинальность категориальных признаков"""
        df = pd.DataFrame({
            'high_card': [f"cat_{i}" for i in range(100)],
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(df)
        
        assert forge._is_fitted is True
    
    def test_constant_columns(self):
        """Константные колонки"""
        df = pd.DataFrame({
            'constant': [1.0] * 100,
            'feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            forge.fit(df)
        
        assert forge._is_fitted is True
    
    def test_fit_with_warnings(self, simple_df, recwarn):
        """Обучение с предупреждениями"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            # Генерируем предупреждение
            warnings.warn("Test warning", UserWarning)
            
            forge.fit(simple_df)
        
        # Проверяем, что предупреждение было поймано
        assert len(recwarn) > 0
        assert any("Test warning" in str(w.message) for w in recwarn)
    
    def test_repeated_fit_transform(self, simple_df):
        """Повторный вызов fit_transform"""
        forge = AutoForge(target="target", verbose=False)
        
        # Создаем mock pipeline с поддержкой len и execute
        mock_pipeline = Mock()
        mock_pipeline.__len__ = Mock(return_value=5)
        mock_pipeline.execute.return_value = Mock(
            success=True,
            errors=[],
            container=DataContainer(data=simple_df, target_column="target")
        )
        
        with patch.object(forge, '_build_pipeline', return_value=mock_pipeline):
            # Первый вызов
            result1 = forge.fit_transform(simple_df)
            
            # Второй вызов (должен переобучиться)
            result2 = forge.fit_transform(simple_df)
        
        assert isinstance(result1, ForgeResult)
        assert isinstance(result2, ForgeResult)
    
    def test_partial_config_override(self):
        """Частичное переопределение конфигурации"""
        forge = AutoForge(
            target="target",
            task="regression",
            balance=False,
            test_size=0.3
        )
        
        assert forge.config.task == TaskType.REGRESSION
        assert forge.config.balance is False
        assert forge.config.test_size == 0.3
        # Остальные параметры должны быть по умолчанию
        assert forge.config.random_state == 42
        assert forge.config.verbose is True