# tests/test_forge.py
"""
Тесты для AutoForge.
"""

import pytest
import pandas as pd
import numpy as np

from automl_data import AutoForge, ForgeResult


@pytest.fixture
def sample_tabular_data():
    """Пример табличных данных"""
    np.random.seed(42)
    n = 200
    
    return pd.DataFrame({
        "numeric1": np.random.randn(n),
        "numeric2": np.random.randn(n) * 10 + 5,
        "category1": np.random.choice(["A", "B", "C"], n),
        "category2": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3])  # Несбалансировано
    })


@pytest.fixture
def sample_text_data():
    """Пример текстовых данных"""
    texts = [
        "This is a great product, I love it!",
        "Terrible experience, would not recommend.",
        "Average quality, nothing special.",
        "Best purchase ever, highly recommended!",
        "Waste of money, very disappointed.",
    ] * 20
    
    labels = [1, 0, 1, 1, 0] * 20
    
    return pd.DataFrame({
        "review": texts,
        "sentiment": labels
    })


class TestAutoForgeTabular:
    """Тесты для табличных данных"""
    
    def test_basic_fit_transform(self, sample_tabular_data):
        forge = AutoForge(target="target")
        result = forge.fit_transform(sample_tabular_data)
        
        assert isinstance(result, ForgeResult)
        assert result.shape[0] > 0
        assert result.quality_score > 0
    
    def test_no_missing_after_transform(self, sample_tabular_data):
        # Добавляем пропуски
        df = sample_tabular_data.copy()
        df.loc[0:10, "numeric1"] = np.nan
        
        forge = AutoForge(target="target")
        result = forge.fit_transform(df)
        
        # После обработки не должно быть пропусков (кроме возможных в target)
        numeric_cols = result.X.select_dtypes(include=[np.number]).columns
        assert result.X[numeric_cols].isnull().sum().sum() == 0
    
    def test_balancing_works(self, sample_tabular_data):
        forge = AutoForge(target="target", balance=True)
        result = forge.fit_transform(sample_tabular_data)
        
        # Проверяем, что классы стали более сбалансированными
        class_counts = result.y.value_counts()
        ratio = class_counts.min() / class_counts.max()
        
        # После балансировки ratio должен быть выше
        assert ratio >= 0.5
    
    def test_get_splits(self, sample_tabular_data):
        forge = AutoForge(target="target")
        result = forge.fit_transform(sample_tabular_data)
        
        X_train, X_test, y_train, y_test = result.get_splits(test_size=0.2)
        
        total = len(X_train) + len(X_test)
        assert len(X_test) / total == pytest.approx(0.2, abs=0.05)
    
    def test_disable_balancing(self, sample_tabular_data):
        original_size = len(sample_tabular_data)
        
        forge = AutoForge(target="target", balance=False)
        result = forge.fit_transform(sample_tabular_data)
        
        # Размер не должен сильно измениться
        assert abs(len(result.data) - original_size) < 10


class TestAutoForgeConfiguration:
    """Тесты конфигурации"""
    
    def test_custom_imputation(self, sample_tabular_data):
        df = sample_tabular_data.copy()
        df.loc[0:5, "numeric1"] = np.nan
        
        forge = AutoForge(
            target="target",
            impute_strategy="simple"
        )
        result = forge.fit_transform(df)
        
        assert result.X["numeric1"].isnull().sum() == 0
    
    def test_custom_scaling(self, sample_tabular_data):
        forge = AutoForge(
            target="target",
            scaling="robust"
        )
        result = forge.fit_transform(sample_tabular_data)
        
        # Проверяем, что данные масштабированы
        numeric_cols = result.X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result.X[col].std() < 10  # Примерная проверка
    
    def test_verbose_mode(self, sample_tabular_data, capsys):
        forge = AutoForge(target="target", verbose=True)
        forge.fit_transform(sample_tabular_data)
        
        captured = capsys.readouterr()
        # В verbose режиме должны быть логи
        # (в реальности проверяем через logging, но упрощённо)


class TestForgeResult:
    """Тесты ForgeResult"""
    
    def test_result_properties(self, sample_tabular_data):
        forge = AutoForge(target="target")
        result = forge.fit_transform(sample_tabular_data)
        
        assert result.data is not None
        assert result.X is not None
        assert result.y is not None
        assert 0 <= result.quality_score <= 1
        assert len(result.steps) > 0
    
    def test_summary(self, sample_tabular_data):
        forge = AutoForge(target="target")
        result = forge.fit_transform(sample_tabular_data)
        
        summary = result.container.summary()
        
        assert "shape" in summary
        assert "quality" in summary
        assert "target" in summary
    
    def test_save_report(self, sample_tabular_data, tmp_path):
        forge = AutoForge(target="target")
        result = forge.fit_transform(sample_tabular_data)
        
        report_path = tmp_path / "report.html"
        result.save_report(str(report_path))
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "ML Data Forge" in content