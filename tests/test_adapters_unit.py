# tests/test_adapters_unit.py
"""
Независимые юнит-тесты для каждого адаптера.

Каждый адаптер тестируется изолированно:
- Создание и инициализация
- fit() 
- transform()
- fit_transform()
- Граничные случаи
- Параметры и конфигурация
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from automl_data.core.container import DataContainer, ProcessingStage
from automl_data.utils.exceptions import ValidationError, NotFittedError


# ==================== FIXTURES ====================

@pytest.fixture
def simple_numeric_df() -> pd.DataFrame:
    """Простой числовой датасет"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.uniform(0, 100, 100),
        'feature_3': np.random.randint(0, 1000, 100).astype(float),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def mixed_types_df() -> pd.DataFrame:
    """Датасет со смешанными типами"""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_1': np.random.randn(100),
        'numeric_2': np.random.uniform(0, 100, 100),
        'category_1': np.random.choice(['A', 'B', 'C'], 100),
        'category_2': np.random.choice(['X', 'Y'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Датасет с пропусками"""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_1': np.random.randn(100),
        'numeric_2': np.random.uniform(0, 100, 100),
        'category_1': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Добавляем пропуски
    df.loc[np.random.choice(100, 15, replace=False), 'numeric_1'] = np.nan
    df.loc[np.random.choice(100, 10, replace=False), 'numeric_2'] = np.nan
    df.loc[np.random.choice(100, 8, replace=False), 'category_1'] = np.nan
    
    return df


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """Датасет с выбросами"""
    np.random.seed(42)
    df = pd.DataFrame({
        'normal': np.random.randn(100),
        'with_outliers': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Добавляем выбросы
    outlier_idx = np.random.choice(100, 10, replace=False)
    df.loc[outlier_idx, 'with_outliers'] = np.random.choice([-100, 100], 10)
    
    return df


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """Несбалансированный датасет"""
    np.random.seed(42)
    
    # 90% класс 0, 10% класс 1
    n_majority = 90
    n_minority = 10
    
    return pd.DataFrame({
        'feature_1': np.random.randn(n_majority + n_minority),
        'feature_2': np.random.uniform(0, 100, n_majority + n_minority),
        'target': [0] * n_majority + [1] * n_minority
    })


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """Датасет с высокой кардинальностью"""
    np.random.seed(42)
    return pd.DataFrame({
        'low_card': np.random.choice(['A', 'B', 'C'], 100),
        'high_card': [f'cat_{i % 50}' for i in range(100)],
        'numeric': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def duplicate_features_df() -> pd.DataFrame:
    """Датасет с дублирующимися признаками"""
    np.random.seed(42)
    base = np.random.randn(100)
    return pd.DataFrame({
        'original': base,
        'duplicate': base.copy(),  # Полный дубликат
        'constant': np.ones(100),  # Константа
        'almost_constant': np.concatenate([np.ones(99), [0]]),
        'target': np.random.choice([0, 1], 100)
    })


def make_container(df: pd.DataFrame, target: str = 'target') -> DataContainer:
    """Хелпер для создания контейнера"""
    return DataContainer(data=df.copy(), target_column=target)


# ==================== BASE ADAPTER TESTS ====================

class TestBaseAdapter:
    """Тесты базового адаптера"""
    
    def test_adapter_not_fitted_raises_error(self, simple_numeric_df):
        """Проверяет, что transform без fit вызывает ошибку"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter(strategy="standard")
        
        with pytest.raises(NotFittedError):
            adapter.transform(container)
    
    def test_adapter_is_fitted_after_fit(self, simple_numeric_df):
        """Проверяет установку флага is_fitted"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter(strategy="standard")
        
        assert not adapter.is_fitted
        
        adapter.fit(container)
        
        assert adapter.is_fitted
    
    def test_fit_returns_self(self, simple_numeric_df):
        """Проверяет, что fit возвращает self для chaining"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter()
        
        result = adapter.fit(container)
        
        assert result is adapter
    
    def test_empty_container_raises_error(self):
        """Проверяет ошибку для пустого контейнера"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        empty_df = pd.DataFrame(columns=['a', 'b', 'target'])
        container = make_container(empty_df)
        
        adapter = ScalingAdapter()
        
        with pytest.raises(ValidationError):
            adapter.fit(container)


# ==================== SCALING ADAPTER TESTS ====================

class TestScalingAdapter:
    """Тесты ScalingAdapter"""
    
    def test_standard_scaling(self, simple_numeric_df):
        """Тест стандартного масштабирования"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter(strategy="standard")
        
        result = adapter.fit_transform(container)
        
        # Проверяем, что числовые колонки масштабированы
        for col in ['feature_1', 'feature_2', 'feature_3']:
            values = result.data[col]
            # После стандартизации: mean ≈ 0, std ≈ 1
            assert abs(values.mean()) < 0.1, f"{col} mean should be ~0"
            assert abs(values.std() - 1) < 0.1, f"{col} std should be ~1"
    
    def test_robust_scaling(self, df_with_outliers):
        """Тест робастного масштабирования"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(df_with_outliers)
        adapter = ScalingAdapter(strategy="robust")
        
        result = adapter.fit_transform(container)
        
        assert adapter._actual_strategy == "robust"
        assert result.data is not None
    
    def test_minmax_scaling(self, simple_numeric_df):
        """Тест MinMax масштабирования"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter(strategy="minmax")
        
        result = adapter.fit_transform(container)
        
        # Проверяем диапазон [0, 1]
        for col in ['feature_1', 'feature_2', 'feature_3']:
            values = result.data[col]
            assert values.min() >= -0.01, f"{col} min should be >= 0"
            assert values.max() <= 1.01, f"{col} max should be <= 1"
    
    def test_auto_strategy_detects_outliers(self, df_with_outliers):
        """Тест автовыбора стратегии при выбросах"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(df_with_outliers)
        adapter = ScalingAdapter(strategy="auto")
        
        adapter.fit(container)
        
        # При выбросах должен выбрать robust
        assert adapter._actual_strategy == "robust"
    
    def test_none_strategy_skips(self, simple_numeric_df):
        """Тест пропуска масштабирования"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        original_values = simple_numeric_df['feature_1'].copy()
        
        adapter = ScalingAdapter(strategy="none")
        result = adapter.fit_transform(container)
        
        # Данные не должны измениться
        pd.testing.assert_series_equal(
            result.data['feature_1'].reset_index(drop=True),
            original_values.reset_index(drop=True),
            check_names=False
        )
    
    def test_target_not_scaled(self, simple_numeric_df):
        """Проверяет, что target не масштабируется"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        original_target = simple_numeric_df['target'].copy()
        
        adapter = ScalingAdapter(strategy="standard")
        result = adapter.fit_transform(container)
        
        # Target должен остаться без изменений
        np.testing.assert_array_equal(
            result.data['target'].values,
            original_target.values
        )
    
    def test_fit_info_populated(self, simple_numeric_df):
        """Проверяет заполнение fit_info"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        adapter = ScalingAdapter(strategy="standard")
        
        adapter.fit(container)
        
        assert "strategy" in adapter._fit_info
        assert "columns_count" in adapter._fit_info


# ==================== IMPUTATION ADAPTER TESTS ====================

class TestImputationAdapter:
    """Тесты ImputationAdapter"""
    
    def test_simple_imputation(self, df_with_missing):
        """Тест простой импьютации"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        missing_before = container.data.isnull().sum().sum()
        
        adapter = ImputationAdapter(strategy="simple")
        result = adapter.fit_transform(container)
        
        missing_after = result.data.isnull().sum().sum()
        
        assert missing_after < missing_before
        assert missing_after == 0, "All missing values should be imputed"
    
    def test_knn_imputation(self, df_with_missing):
        """Тест KNN импьютации"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        
        adapter = ImputationAdapter(strategy="knn", n_neighbors=3)
        result = adapter.fit_transform(container)
        
        # Проверяем только числовые колонки (KNN работает с ними)
        numeric_missing = result.data[['numeric_1', 'numeric_2']].isnull().sum().sum()
        assert numeric_missing == 0
    
    def test_auto_strategy_selection(self, df_with_missing):
        """Тест автовыбора стратегии"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        
        adapter = ImputationAdapter(strategy="auto")
        adapter.fit(container)
        
        # Должна выбраться какая-то стратегия
        assert adapter._actual_strategy in ["simple", "knn", "iterative"]
    
    def test_categorical_imputation(self, df_with_missing):
        """Тест импьютации категориальных"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        
        adapter = ImputationAdapter(strategy="simple")
        result = adapter.fit_transform(container)
        
        # Категориальные пропуски должны быть заполнены
        assert result.data['category_1'].isnull().sum() == 0
    
    def test_none_strategy_skips(self, df_with_missing):
        """Тест пропуска импьютации"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        missing_before = container.data.isnull().sum().sum()
        
        adapter = ImputationAdapter(strategy="none")
        result = adapter.fit_transform(container)
        
        missing_after = result.data.isnull().sum().sum()
        
        assert missing_after == missing_before
    
    def test_target_not_imputed(self, df_with_missing):
        """Проверяет, что target не изменяется"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        # Добавляем пропуск в target
        df = df_with_missing.copy()
        original_target = df['target'].copy()
        
        container = make_container(df)
        
        adapter = ImputationAdapter(strategy="simple")
        result = adapter.fit_transform(container)
        
        # Target должен остаться без изменений
        np.testing.assert_array_equal(
            result.data['target'].values,
            original_target.values
        )


# ==================== OUTLIER ADAPTER TESTS ====================

class TestOutlierAdapter:
    """Тесты OutlierAdapter"""
    
    @pytest.fixture(autouse=True)
    def check_pyod(self):
        """Пропускаем если нет pyod"""
        pytest.importorskip("pyod")
    
    def test_clip_action(self, df_with_outliers):
        """Тест отсечения выбросов"""
        from automl_data.adapters.outliers import OutlierAdapter
        
        container = make_container(df_with_outliers)
        original_max = container.data['with_outliers'].abs().max()
        
        adapter = OutlierAdapter(method="auto", action="clip")
        result = adapter.fit_transform(container)
        
        new_max = result.data['with_outliers'].abs().max()
        
        # Максимальное значение должно уменьшиться
        assert new_max <= original_max
    
    def test_remove_action(self, df_with_outliers):
        """Тест удаления выбросов"""
        from automl_data.adapters.outliers import OutlierAdapter
        
        container = make_container(df_with_outliers)
        original_size = len(container.data)
        
        adapter = OutlierAdapter(method="auto", action="remove", contamination=0.1)
        result = adapter.fit_transform(container)
        
        new_size = len(result.data)
        
        # Размер должен уменьшиться
        assert new_size <= original_size
    
    def test_flag_action(self, df_with_outliers):
        """Тест флагирования выбросов"""
        from automl_data.adapters.outliers import OutlierAdapter
        
        container = make_container(df_with_outliers)
        
        adapter = OutlierAdapter(method="auto", action="flag")
        result = adapter.fit_transform(container)
        
        # Должна появиться колонка с флагом
        assert "_is_outlier" in result.data.columns
        assert result.data["_is_outlier"].dtype == bool
    
    def test_iforest_method(self, df_with_outliers):
        """Тест Isolation Forest"""
        from automl_data.adapters.outliers import OutlierAdapter
        
        container = make_container(df_with_outliers)
        
        adapter = OutlierAdapter(method="iforest", action="clip")
        result = adapter.fit_transform(container)
        
        assert adapter._fit_info.get("detector") == "IForest"


# ==================== ENCODING ADAPTER TESTS ====================

class TestEncodingAdapter:
    """Тесты EncodingAdapter"""
    
    @pytest.fixture(autouse=True)
    def check_category_encoders(self):
        """Пропускаем если нет category_encoders"""
        pytest.importorskip("category_encoders")
    
    def test_onehot_encoding(self, mixed_types_df):
        """Тест One-Hot кодирования"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        container = make_container(mixed_types_df)
        original_cols = len(container.data.columns)
        
        adapter = EncodingAdapter(strategy="onehot", max_onehot_cardinality=10)
        result = adapter.fit_transform(container)
        
        # Колонок должно стать больше (one-hot расширяет)
        assert len(result.data.columns) > original_cols
        
        # Исходные категориальные должны исчезнуть
        assert 'category_1' not in result.data.columns
        assert 'category_2' not in result.data.columns
    
    def test_ordinal_encoding(self, mixed_types_df):
        """Тест Ordinal кодирования"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        container = make_container(mixed_types_df)
        
        adapter = EncodingAdapter(strategy="ordinal")
        result = adapter.fit_transform(container)
        
        # Все колонки должны быть числовыми
        for col in result.data.columns:
            if col != 'target':
                assert pd.api.types.is_numeric_dtype(result.data[col]), \
                    f"Column {col} should be numeric"
    
    def test_target_encoding(self, mixed_types_df):
        """Тест Target кодирования"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        container = make_container(mixed_types_df)
        
        adapter = EncodingAdapter(strategy="target", target_column="target")
        result = adapter.fit_transform(container)
        
        # Все колонки должны быть числовыми
        for col in result.data.columns:
            assert pd.api.types.is_numeric_dtype(result.data[col])
    
    def test_auto_strategy_high_cardinality(self, high_cardinality_df):
        """Тест автовыбора для высокой кардинальности"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        container = make_container(high_cardinality_df)
        
        adapter = EncodingAdapter(strategy="auto", max_onehot_cardinality=10)
        adapter.fit(container)
        
        # high_card имеет 50 уникальных значений -> не должен быть onehot
        strategy = adapter._strategies.get('high_card', '')
        assert strategy != "onehot"
    
    def test_target_preserved(self, mixed_types_df):
        """Проверяет сохранение target"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        container = make_container(mixed_types_df)
        original_target = mixed_types_df['target'].copy()
        
        adapter = EncodingAdapter(strategy="onehot")
        result = adapter.fit_transform(container)
        
        assert 'target' in result.data.columns
        np.testing.assert_array_equal(
            result.data['target'].values,
            original_target.values
        )


# ==================== FEATURE CLEANER ADAPTER TESTS ====================

class TestFeatureCleanerAdapter:
    """Тесты FeatureCleanerAdapter"""
    
    def test_removes_constant_features(self, duplicate_features_df):
        """Тест удаления константных признаков"""
        from automl_data.adapters.feature_cleaner import FeatureCleanerAdapter
        
        container = make_container(duplicate_features_df)
        
        adapter = FeatureCleanerAdapter(remove_duplicates=False)
        result = adapter.fit_transform(container)
        
        assert 'constant' not in result.data.columns
    
    def test_removes_duplicate_features(self, duplicate_features_df):
        """Тест удаления дубликатов признаков"""
        from automl_data.adapters.feature_cleaner import FeatureCleanerAdapter
        
        container = make_container(duplicate_features_df)
        
        adapter = FeatureCleanerAdapter(remove_duplicates=True)
        result = adapter.fit_transform(container)
        
        # original или duplicate должен остаться, но не оба
        has_original = 'original' in result.data.columns
        has_duplicate = 'duplicate' in result.data.columns
        
        # Один из них должен быть удалён
        assert not (has_original and has_duplicate), \
            "Duplicate should be removed"
    
    def test_removes_high_missing(self):
        """Тест удаления признаков с высоким % пропусков"""
        from automl_data.adapters.feature_cleaner import FeatureCleanerAdapter
        
        df = pd.DataFrame({
            'good': np.random.randn(100),
            'bad': np.concatenate([np.random.randn(5), [np.nan] * 95]),
            'target': np.random.choice([0, 1], 100)
        })
        
        container = make_container(df)
        
        adapter = FeatureCleanerAdapter(max_missing_ratio=0.5)
        result = adapter.fit_transform(container)
        
        assert 'bad' not in result.data.columns
        assert 'good' in result.data.columns
    
    def test_removes_id_columns(self):
        """Тест удаления ID колонок"""
        from automl_data.adapters.feature_cleaner import FeatureCleanerAdapter
        
        df = pd.DataFrame({
            'id': range(100),
            'user_id': [f'user_{i}' for i in range(100)],
            'feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        container = make_container(df)
        
        adapter = FeatureCleanerAdapter(remove_id_columns=True)
        result = adapter.fit_transform(container)
        
        assert 'id' not in result.data.columns
        assert 'user_id' not in result.data.columns
        assert 'feature' in result.data.columns
    
    def test_target_preserved(self, duplicate_features_df):
        """Проверяет сохранение target"""
        from automl_data.adapters.feature_cleaner import FeatureCleanerAdapter
        
        container = make_container(duplicate_features_df)
        
        adapter = FeatureCleanerAdapter()
        result = adapter.fit_transform(container)
        
        assert 'target' in result.data.columns


# ==================== BALANCING ADAPTER TESTS ====================

class TestBalancingAdapter:
    """Тесты BalancingAdapter"""
    
    @pytest.fixture(autouse=True)
    def check_imblearn(self):
        """Пропускаем если нет imblearn"""
        pytest.importorskip("imblearn")
    
    def test_smote_increases_minority(self, imbalanced_df):
        """Тест увеличения миноритарного класса через SMOTE"""
        from automl_data.adapters.balancing import BalancingAdapter
        
        container = make_container(imbalanced_df)
        original_minority = (container.data['target'] == 1).sum()
        
        adapter = BalancingAdapter(
            strategy="smote",
            target_column="target",
            imbalance_threshold=0.3
        )
        result = adapter.fit_transform(container)
        
        new_minority = (result.data['target'] == 1).sum()
        
        assert new_minority > original_minority
    
    def test_random_oversampling(self, imbalanced_df):
        """Тест случайного oversampling"""
        from automl_data.adapters.balancing import BalancingAdapter
        
        container = make_container(imbalanced_df)
        original_size = len(container.data)
        
        adapter = BalancingAdapter(
            strategy="random_over",
            target_column="target",
            imbalance_threshold=0.3
        )
        result = adapter.fit_transform(container)
        
        assert len(result.data) > original_size
    
    def test_skips_balanced_data(self):
        """Тест пропуска сбалансированных данных"""
        from automl_data.adapters.balancing import BalancingAdapter
        
        # Сбалансированный датасет
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': [0] * 50 + [1] * 50
        })
        
        container = make_container(df)
        original_size = len(container.data)
        
        adapter = BalancingAdapter(
            strategy="smote",
            target_column="target",
            imbalance_threshold=0.3
        )
        result = adapter.fit_transform(container)
        
        # Размер не должен измениться
        assert len(result.data) == original_size
    
    def test_preserves_feature_types(self, imbalanced_df):
        """Проверяет сохранение типов признаков"""
        from automl_data.adapters.balancing import BalancingAdapter
        
        container = make_container(imbalanced_df)
        
        adapter = BalancingAdapter(
            strategy="random_over",
            target_column="target"
        )
        result = adapter.fit_transform(container)
        
        # Числовые признаки должны остаться числовыми
        assert pd.api.types.is_numeric_dtype(result.data['feature_1'])
        assert pd.api.types.is_numeric_dtype(result.data['feature_2'])


# ==================== PROFILER ADAPTER TESTS ====================

class TestProfilerAdapter:
    """Тесты ProfilerAdapter"""
    
    def test_basic_profiling(self, simple_numeric_df):
        """Тест базового профилирования"""
        from automl_data.adapters.profiling import ProfilerAdapter
        
        container = make_container(simple_numeric_df)
        
        adapter = ProfilerAdapter(minimal=True)
        result = adapter.fit_transform(container)
        
        # Должен заполнить профиль
        assert result.profile is not None
        assert len(result.profile) > 0
    
    def test_detects_missing(self, df_with_missing):
        """Тест обнаружения пропусков"""
        from automl_data.adapters.profiling import ProfilerAdapter
        
        container = make_container(df_with_missing)
        
        adapter = ProfilerAdapter(minimal=True)
        result = adapter.fit_transform(container)
        
        # Должен обнаружить пропуски
        profile = result.profile
        assert "missing_percent" in profile or "missing" in str(profile).lower()


# ==================== EDGE CASES ====================

class TestAdapterEdgeCases:
    """Тесты граничных случаев"""
    
    def test_single_row_dataframe(self):
        """Тест с одной строкой"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        df = pd.DataFrame({
            'feature': [1.0],
            'target': [0]
        })
        
        container = make_container(df)
        adapter = ScalingAdapter(strategy="standard")
        
        # Не должно падать
        result = adapter.fit_transform(container)
        assert len(result.data) == 1
    
    def test_no_numeric_columns(self):
        """Тест без числовых колонок"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 33 + ['A'],
            'cat2': ['X', 'Y'] * 50,
            'target': [0, 1] * 50
        })
        
        container = make_container(df)
        adapter = ScalingAdapter(strategy="standard")
        
        # Должно пропустить масштабирование
        result = adapter.fit_transform(container)
        assert result.data is not None
    
    def test_no_categorical_columns(self):
        """Тест без категориальных колонок"""
        from automl_data.adapters.encoding import EncodingAdapter
        
        pytest.importorskip("category_encoders")
        
        df = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        container = make_container(df)
        adapter = EncodingAdapter(strategy="onehot")
        
        # Должно пропустить кодирование
        result = adapter.fit_transform(container)
        assert result.data.shape == df.shape


# ==================== RECOMMENDATIONS TESTS ====================

class TestAdapterRecommendations:
    """Тесты рекомендаций"""
    
    def test_scaling_adds_recommendation(self, simple_numeric_df):
        """Проверяет добавление рекомендаций"""
        from automl_data.adapters.scaling import ScalingAdapter
        
        container = make_container(simple_numeric_df)
        
        adapter = ScalingAdapter(strategy="standard")
        result = adapter.fit_transform(container)
        
        assert len(result.recommendations) > 0
        
        rec = result.recommendations[-1]
        assert rec.get('type') == 'scaling'
    
    def test_imputation_adds_recommendation(self, df_with_missing):
        """Проверяет рекомендации импьютации"""
        from automl_data.adapters.imputation import ImputationAdapter
        
        container = make_container(df_with_missing)
        
        adapter = ImputationAdapter(strategy="simple")
        result = adapter.fit_transform(container)
        
        recs = [r for r in result.recommendations if r.get('type') == 'imputation']
        assert len(recs) > 0