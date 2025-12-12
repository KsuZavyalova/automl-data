# tests/test_adapters.py
"""
Тесты для адаптеров.
"""

import pytest
import pandas as pd
import numpy as np

from automl_data.core.container import DataContainer
from automl_data.adapters.encoding import EncodingAdapter
from automl_data.adapters.outliers import OutlierAdapter
from automl_data.adapters.balancing import BalancingAdapter


@pytest.fixture
def numeric_container():
    """Контейнер с числовыми данными"""
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": np.random.randn(100) * 10,
        "target": np.random.choice([0, 1], 100)
    })
    # Добавляем пропуски
    df.loc[0:10, "a"] = np.nan
    
    return DataContainer(data=df, target_column="target")


@pytest.fixture
def categorical_container():
    """Контейнер с категориальными данными"""
    df = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], 100),
        "cat2": np.random.choice(["X", "Y"], 100),
        "num": np.random.randn(100),
        "target": np.random.choice([0, 1], 100)
    })
    return DataContainer(data=df, target_column="target")


class TestEncodingAdapter:
    """Тесты EncodingAdapter"""
    
    def test_onehot_encoding(self, categorical_container):
        adapter = EncodingAdapter(strategy="onehot")
        result = adapter.fit_transform(categorical_container)
        
        # Оригинальных категориальных колонок быть не должно
        assert "cat1" not in result.data.columns
        
        # Должны появиться one-hot колонки
        onehot_cols = [c for c in result.data.columns if "cat1" in c]
        assert len(onehot_cols) > 0
    
    def test_auto_encoding(self, categorical_container):
        adapter = EncodingAdapter(strategy="auto")
        result = adapter.fit_transform(categorical_container)
        
        # После кодирования все колонки (кроме возможных новых) должны быть числовыми
        numeric_ratio = len(result.data.select_dtypes(include=[np.number]).columns) / len(result.data.columns)
        assert numeric_ratio > 0.5


class TestOutlierAdapter:
    """Тесты OutlierAdapter"""
    
    def test_outlier_detection(self):
        # Создаём данные с явными выбросами
        np.random.seed(42)
        df = pd.DataFrame({
            "a": list(np.random.randn(95)) + [100, -100, 50, -50, 200],
            "target": [0] * 100
        })
        container = DataContainer(data=df, target_column="target")
        
        adapter = OutlierAdapter(method="iforest", action="flag")
        result = adapter.fit_transform(container)
        
        assert "_is_outlier" in result.data.columns
        assert result.data["_is_outlier"].sum() > 0
    
    def test_clip_action(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "a": list(np.random.randn(95)) + [100, -100, 50, -50, 200],
            "target": [0] * 100
        })
        container = DataContainer(data=df, target_column="target")
        
        adapter = OutlierAdapter(method="iforest", action="clip")
        result = adapter.fit_transform(container)
        
        # После clipping экстремальных значений быть не должно
        assert result.data["a"].max() < 100


class TestBalancingAdapter:
    """Тесты BalancingAdapter"""
    
    def test_smote_balancing(self):
        # Сильно несбалансированные данные
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "target": [0] * 90 + [1] * 10
        })
        container = DataContainer(data=df, target_column="target")
        
        adapter = BalancingAdapter(strategy="smote")
        result = adapter.fit_transform(container)
        
        # После балансировки классы должны быть сбалансированы
        class_counts = result.data["target"].value_counts()
        ratio = class_counts.min() / class_counts.max()
        
        assert ratio > 0.5
    
    def test_no_balancing_needed(self):
        # Уже сбалансированные данные
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "target": [0] * 50 + [1] * 50
        })
        container = DataContainer(data=df, target_column="target")
        
        original_size = len(df)
        
        adapter = BalancingAdapter(strategy="auto", imbalance_threshold=0.3)
        result = adapter.fit_transform(container)
        
        # Размер не должен измениться
        assert len(result.data) == original_size