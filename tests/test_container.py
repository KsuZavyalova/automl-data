# tests/test_container.py
"""
Тесты для DataContainer.
"""

import pytest
import pandas as pd
import numpy as np

from automl_data.core.container import (
    DataContainer, 
    DataType, 
    ProcessingStage,
    ProcessingStep
)


class TestDataContainerCreation:
    """Тесты создания контейнера"""
    
    def test_create_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        container = DataContainer(data=df)
        
        assert container.shape == (3, 2)
        assert container.stage == ProcessingStage.RAW
        assert len(container) == 3
    
    def test_create_with_target(self):
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        container = DataContainer(data=df, target_column="target")
        
        assert container.target_column == "target"
        assert container.y is not None
        assert list(container.y) == [0, 1, 0]
    
    def test_create_with_invalid_data(self):
        with pytest.raises(TypeError):
            DataContainer(data=[1, 2, 3])
    
    def test_auto_detect_text_type(self):
        df = pd.DataFrame({
            "text": ["This is a long text " * 10] * 5,
            "label": [0, 1, 0, 1, 0]
        })
        container = DataContainer(data=df)
        
        assert container.data_type == DataType.TEXT
        assert container.text_column == "text"


class TestDataContainerOperators:
    """Тесты перегрузки операторов"""
    
    def test_len(self):
        df = pd.DataFrame({"a": range(100)})
        container = DataContainer(data=df)
        
        assert len(container) == 100
    
    def test_bool(self):
        df_empty = pd.DataFrame()
        df_full = pd.DataFrame({"a": [1]})
        
        assert not bool(DataContainer(data=df_empty))
        assert bool(DataContainer(data=df_full))
    
    def test_contains(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        container = DataContainer(data=df)
        
        assert "a" in container
        assert "c" not in container
    
    def test_getitem_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        container = DataContainer(data=df)
        
        result = container["a"]
        assert list(result) == [1, 2, 3]
    
    def test_getitem_slice(self):
        df = pd.DataFrame({"a": range(10)})
        container = DataContainer(data=df)
        
        subset = container[:5]
        assert len(subset) == 5
        assert isinstance(subset, DataContainer)
    
    def test_add_containers(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        
        c1 = DataContainer(data=df1)
        c2 = DataContainer(data=df2)
        
        combined = c1 + c2
        assert len(combined) == 4
        assert list(combined["a"]) == [1, 2, 3, 4]
    
    def test_iter(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        container = DataContainer(data=df)
        
        rows = list(container)
        assert len(rows) == 3


class TestDataContainerMethods:
    """Тесты методов"""
    
    def test_clone(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        container = DataContainer(data=df)
        
        clone = container.clone()
        clone["a"] = [4, 5, 6]
        
        # Оригинал не изменился
        assert list(container["a"]) == [1, 2, 3]
    
    def test_split(self):
        df = pd.DataFrame({"a": range(100), "label": [0, 1] * 50})
        container = DataContainer(data=df, target_column="label")
        
        train, test = container.split(train_ratio=0.8)
        
        assert len(train) == 80
        assert len(test) == 20
    
    def test_filter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        container = DataContainer(data=df)
        
        filtered = container.filter(container["a"] > 2)
        assert len(filtered) == 3
    
    def test_sample(self):
        df = pd.DataFrame({"a": range(100)})
        container = DataContainer(data=df)
        
        sampled = container.sample(n=10)
        assert len(sampled) == 10
    
    def test_summary(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        container = DataContainer(data=df, target_column="b")
        
        summary = container.summary()
        
        assert summary["shape"] == (3, 2)
        assert summary["target"] == "b"


class TestDataContainerProperties:
    """Тесты свойств"""
    
    def test_numeric_columns(self):
        df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
        container = DataContainer(data=df)
        
        assert container.numeric_columns == ["num"]
    
    def test_categorical_columns(self):
        df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
        container = DataContainer(data=df)
        
        assert container.categorical_columns == ["cat"]
    
    def test_X_y_properties(self):
        df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "target": [0, 1]})
        container = DataContainer(data=df, target_column="target")
        
        assert list(container.X.columns) == ["f1", "f2"]
        assert list(container.y) == [0, 1]
    
    def test_class_distribution(self):
        df = pd.DataFrame({"target": [0, 0, 0, 1, 1]})
        container = DataContainer(data=df, target_column="target")
        
        dist = container.class_distribution
        assert dist[0] == 3
        assert dist[1] == 2
    
    def test_is_imbalanced(self):
        df_balanced = pd.DataFrame({"target": [0, 0, 1, 1]})
        df_imbalanced = pd.DataFrame({"target": [0] * 90 + [1] * 10})
        
        c1 = DataContainer(data=df_balanced, target_column="target")
        c2 = DataContainer(data=df_imbalanced, target_column="target")
        
        assert not c1.is_imbalanced
        assert c2.is_imbalanced